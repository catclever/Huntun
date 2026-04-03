import numpy as np
import pandas as pd

import json
import os
import glob
import re
import queue
import threading
from typing import List, Tuple, Dict, Any

class MultiEmbDataLoader:
    """
    Dataloader that handles:
    1. Reading text chunks from Parquet (in-memory, fast enough for 7M strings).
    2. Zero-copy / mmap loading of 4 massive .npy embedding files.
    3. Dynamic tokenization & padding.
    4. Interrupt / Resume Checkpointing (Saving the exact shuffled order).
    """
    def __init__(self, 
                 parquet_path: str,
                 emb_paths: List[str],
                 tokenizer,
                 batch_size: int = 256,
                 max_seq_len: int = 512,
                 shuffle: bool = True,
                 seed: int = 42,
                 backend: str = 'mlx'):
        
        self.backend = backend
        
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.tokenizer = tokenizer
        
        # 1. Load Text Data
        print(f"Loading Text Parquet from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        # Explode the lists of chunks into a flat list of 7.6M strings
        # dropna() just in case
        self.text_chunks = df['chunks'].explode().dropna().tolist()
        self.total_samples = len(self.text_chunks)
        print(f"Loaded {self.total_samples} text chunks into memory.")
        del df # free up memory of the original dataframe
        
        # 2. Memory-Map the Numpy Embeddings
        print("Memory-mapping embedding arrays...")
        self.embs = []
        for path in emb_paths:
            # mmap_mode='r' completely avoids loading the massive files into RAM
            arr = np.load(path, mmap_mode='r')
            assert arr.shape[0] == self.total_samples, f"Shape mismatch in {path}: {arr.shape[0]} != {self.total_samples}"
            self.embs.append(arr)
            
        print("Embeddings mapped successfully.")
        
        # 3. Epoch & Batch tracking
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        self.batch_idx = 0
        self.indices = np.arange(self.total_samples)
        
        if self.shuffle:
            self.rng.shuffle(self.indices)
            
        self.num_batches = int(np.ceil(self.total_samples / self.batch_size))
        
    def __iter__(self):
        return self
        
    def __next__(self) -> Tuple[Any, List[Any], Any]:
        if self.batch_idx >= self.num_batches:
            # End of Epoch
            self.current_epoch += 1
            self.batch_idx = 0
            if self.shuffle:
                self.rng.shuffle(self.indices)
            raise StopIteration
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        
        # 1. Fetch Texts
        batch_texts = [self.text_chunks[i] for i in batch_indices]
        
        # 2. Tokenize and Pad dynamically using Custom CharTokenizer
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch_texts]
        max_len = max(len(seq) for seq in encoded_list)
        
        padded_ids = []
        masks = []
        for seq in encoded_list:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
            
        # 3. Fetch Embeddings via Mmap
        # This triggers Disk I/O precisely for these specific indices
        if self.backend == 'torch':
            import torch
            token_inputs = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask = torch.tensor(masks, dtype=torch.float32)
            batch_embs = [torch.from_numpy(np.array(arr[batch_indices])).float() for arr in self.embs]
        else:
            import mlx.core as mx
            token_inputs = mx.array(padded_ids)
            attention_mask = mx.array(masks)
            batch_embs = [mx.array(arr[batch_indices]) for arr in self.embs]
        
        self.batch_idx += 1
        
        return token_inputs, batch_embs, attention_mask

    # --- Checkpointing / Interrupt Resume Support ---
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the internal state for resuming."""
        return {
            "current_epoch": self.current_epoch,
            "batch_idx": self.batch_idx,
            "indices": self.indices.tolist() # Safe to serialize exactly how it was shuffled
        }
        
    def load_state_dict(self, state: Dict[str, Any]):
        """Restores the exact shuffle and position."""
        self.current_epoch = state["current_epoch"]
        self.batch_idx = state["batch_idx"]
        self.indices = np.array(state["indices"])
        print(f"Resumed Dataloader from Epoch {self.current_epoch}, Batch {self.batch_idx}/{self.num_batches}")

class ChunkedNpzDataLoader:
    """
    专门适配从 ModelScope (如 catclever/emb_npy) 拉取的 `.npz` 切片矩阵设计的双层乱序 DataLoader。
    机制：
    1. 不吃内存：只在内存驻留"当前正在训练的任务块"的特征（比如每次只把某个 50w 条特征解压进 RAM）。
    2. 双层乱序 (Double Shuffle)：每个 Epoch 开始时打乱所有 Chunk 文件的流转顺序；在读取单个 Chunk 时，在 Chunk 内部将其局部 batch 完全乱序。
    """
    def __init__(self, 
                 parquet_path: str,
                 models: List[str], # 对应你的模型标识，例如 ["bge", "qwen", "xiaobu"]
                 tokenizer,
                 ms_repo_id: str = None, 
                 local_npz_dir: str = None,
                 batch_size: int = 256,
                 max_seq_len: int = 512,
                 shuffle: bool = True,
                 seed: int = 42,
                 backend: str = 'mlx'):
        
        self.backend = backend
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.tokenizer = tokenizer
        self.models = models
        
        # 1. 自动挂载魔搭数据集或加载本地路径
        if ms_repo_id:
            try:
                from modelscope import snapshot_download
                print(f"[DataLoader] 正在从 ModelScope 极速同步数据分块库: {ms_repo_id}...")
                self.npz_dir = snapshot_download(ms_repo_id, repo_type="dataset")
                print(f"[DataLoader] 远端分块库拉取/命中本地缓存完毕: {self.npz_dir}")
            except ImportError:
                raise RuntimeError("请安装 modelscope: pip install modelscope")
        else:
            self.npz_dir = local_npz_dir
            
        if not self.npz_dir or not os.path.isdir(self.npz_dir):
            raise ValueError("必须提供 ms_repo_id 或有效的 local_npz_dir")

        # 2. 从分块文件中提取跨界物理坐标。
        # 假设文件名格式如：bge_chunk_0000000_0500000.npz
        # 取第一个基准模型（比如 models[0]）作为标杆去探测坐标
        base_model = self.models[0]
        pattern = os.path.join(self.npz_dir, f"{base_model}*chunk_*.npz")
        chunk_files = sorted(glob.glob(pattern))
        
        if not chunk_files:
            raise FileNotFoundError(f"[DataLoader] 在 {self.npz_dir} 没有找到类似 {base_model}_chunk_***.npz 的文件！")
        
        self.chunk_bounds = []
        for f in chunk_files:
            fname = os.path.basename(f)
            # 兼容带有或不带 model 前缀的文件名格式
            match = re.search(r'chunk_(\d+)_(\d+)\.npz', fname)
            if match:
                self.chunk_bounds.append((int(match.group(1)), int(match.group(2))))
                
        # 以 start_idx 排序确保顺序正确
        self.chunk_bounds.sort(key=lambda x: x[0])
        print(f"[DataLoader] 探明分布！共扫出 {len(self.chunk_bounds)} 个分块。")

        # 3. 提取对应的全体文本数据 (保持在内存，纯文本极小)
        print(f"[DataLoader] 正在装载基准文本体系: {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        self.text_chunks = df['chunks'].explode().dropna().tolist()
        self.total_samples = len(self.text_chunks)
        print(f"[DataLoader] 成功吃进文本池: {self.total_samples} 条")
        del df
        
        # 4. 统计与 Epoch 控制变量
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        
        # 宏观控制表 (Chunks)
        self.macro_chunk_indices = np.arange(len(self.chunk_bounds))
        if self.shuffle:
             self.rng.shuffle(self.macro_chunk_indices)
             
        self.active_macro_idx_ptr = 0 # 当前正训练到第几个 chunk
        
        # 微观控制表 (内部流)
        self.active_chunk_embs = None
        self.active_micro_indices = []
        self.active_micro_ptr = 0
        
        # 异步流水线 (Async Prefetch)
        self.prefetch_queue = queue.Queue(maxsize=1) # 内存里永远只多塞一块，极限控流
        self.stop_event = threading.Event()
        
        # 启动后台工人开始搬砖
        self._start_prefetching(start_macro_ptr=0)
        self._pop_next_chunk()

    def _start_prefetching(self, start_macro_ptr):
        """挂起旧线程，启动全新后台线程，从指定的 macro_ptr 开始默默搬取"""
        self.stop_event.set()
        
        # 清空队列残骸
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break
                
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(start_macro_ptr,), daemon=True)
        self.prefetch_thread.start()
        
    def _prefetch_worker(self, start_macro_ptr):
        """高强度纯后台下载/解压打工仔，永远走在 GPU 的前面"""
        worker_macro_ptr = start_macro_ptr
        while not self.stop_event.is_set():
            if worker_macro_ptr >= len(self.macro_chunk_indices):
                # 提示时代结束
                self.prefetch_queue.put(None)
                return

            chunk_idx = self.macro_chunk_indices[worker_macro_ptr]
            start_idx, end_idx = self.chunk_bounds[chunk_idx]
            
            try:
                # print(f"\n[Backend Worker] 悄悄开始向深渊抓取并解压下一块... [{start_idx}:{end_idx}]")
                active_chunk_embs = []
                for model_name in self.models:
                    cand_name = f"{model_name}_chunk_{start_idx:07d}_{end_idx:07d}.npz"
                    path = os.path.join(self.npz_dir, cand_name)
                    if not os.path.exists(path):
                         raise FileNotFoundError(f"后台打工仔抓取失败！目标块缺失：{path}")
                    
                    arr = np.load(path)['features']
                    active_chunk_embs.append(arr)
                    
                cur_chunk_size = end_idx - start_idx
                active_micro_indices = np.arange(cur_chunk_size)
                
                if self.shuffle:
                    # 构建独立的后台局部无序宇宙
                    rng = np.random.default_rng(self.seed + worker_macro_ptr + self.current_epoch)
                    rng.shuffle(active_micro_indices)
                    
                payload = {
                    "embs": active_chunk_embs,
                    "micro_indices": active_micro_indices,
                    "global_start": start_idx,
                    "macro_idx_ptr": worker_macro_ptr
                }
                # 这里会阻塞！直到上一块被 GPU 吃光
                self.prefetch_queue.put(payload)
                worker_macro_ptr += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.prefetch_queue.put(e)
                return

    def _pop_next_chunk(self):
        """主线程调用的吸星大法，把工人装好的箱子拿走"""
        payload = self.prefetch_queue.get()
        if payload is None:
            raise StopIteration
        if isinstance(payload, Exception):
            raise payload
            
        self.active_chunk_embs = payload["embs"]
        self.active_micro_indices = payload["micro_indices"]
        self.active_global_start = payload["global_start"]
        self.active_macro_idx_ptr = payload["macro_idx_ptr"]
        self.active_micro_ptr = 0
        print(f"\n[DataLoader] 💥 无缝吸入后台新缓存块 (全局起点: {self.active_global_start}) - 纯异步无卡顿！")

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Any, List[Any], Any]:
        # 如果当前 Chunk 已经被彻底吸干了，无缝提取下一个
        if self.active_micro_ptr >= len(self.active_micro_indices):
            try:
                self._pop_next_chunk()
            except StopIteration:
                # 一个 Epoch 完结！
                self.current_epoch += 1
                if self.shuffle:
                    self.rng.shuffle(self.macro_chunk_indices)
                # 重新使唤工人从头开始搬砖
                self._start_prefetching(start_macro_ptr=0)
                self._pop_next_chunk()  # 直接抓下一个 Epoch 的第一块
                raise StopIteration
            
        fetch_len = min(self.batch_size, len(self.active_micro_indices) - self.active_micro_ptr)
        batch_local_indices = self.active_micro_indices[self.active_micro_ptr : self.active_micro_ptr + fetch_len]
        self.active_micro_ptr += fetch_len
        
        # 1. 映射回全局索引以抓取物理对应的文本序列
        batch_global_indices = self.active_global_start + batch_local_indices
        batch_texts = [self.text_chunks[i] for i in batch_global_indices]
        
        # 2. Tokenize & Pad (与旧版完全一致)
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch_texts]
        max_len = max(max(len(seq) for seq in encoded_list), 1)
        
        padded_ids = []
        masks = []
        for seq in encoded_list:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
            
        # 3. 局域拷贝内存数据构造 Tensor/Array
        if self.backend == 'torch':
            import torch
            token_inputs = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask = torch.tensor(masks, dtype=torch.float32)
            batch_embs = [torch.from_numpy(arr[batch_local_indices].copy()).float() for arr in self.active_chunk_embs]
        else:
            import mlx.core as mx
            token_inputs = mx.array(padded_ids)
            attention_mask = mx.array(masks)
            batch_embs = [mx.array(arr[batch_local_indices]) for arr in self.active_chunk_embs]
            
        return token_inputs, batch_embs, attention_mask

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_epoch": self.current_epoch,
            "active_macro_idx_ptr": self.active_macro_idx_ptr,
            "macro_chunk_indices": self.macro_chunk_indices.tolist()
        }
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.current_epoch = state["current_epoch"]
        self.active_macro_idx_ptr = state["active_macro_idx_ptr"]
        self.macro_chunk_indices = np.array(state["macro_chunk_indices"])
        print(f"[DataLoader] 🔌 自断点恢复：正处于 Epoch {self.current_epoch}, 宏观 Chunk 指针 {self.active_macro_idx_ptr}/{len(self.macro_chunk_indices)}")
        
        # 断点恢复的究极魔法：命令后台线程清空当前一切，直接从历史断点重新满血搬取！
        self._start_prefetching(start_macro_ptr=self.active_macro_idx_ptr)
        self._pop_next_chunk()
    """
    Dataloader specifically for Phase 1 (Mamba Spatiotemporal Dynamics).
    It pulls *contiguous entire documents* dynamically instead of fixed arbitrary slices,
    to ensure Mamba learns the true, uncorrupted semantic arc of an article.
    """
    def __init__(self, 
                 parquet_path: str,
                 emb_paths: List[str],
                 batch_size: int = 16,
                 max_episode_len: int = None,
                 seed: int = 42,
                 backend: str = 'mlx'):
        
        self.backend = backend
        
        self.batch_size = batch_size
        self.max_episode_len = max_episode_len
        
        # 1. Load exact document boundaries to avoid crossing them!
        print("Reading Document Chunk Counts...")
        df = pd.read_parquet(parquet_path, columns=['chunk_count'])
        self.chunk_counts = df['chunk_count'].values
        
        # 2. Cumulative sum gives us the exact starting index for each document in the giant mmapped arrays
        self.doc_start_indices = np.concatenate(([0], np.cumsum(self.chunk_counts)[:-1]))
        self.total_docs = len(self.chunk_counts)
        print(f"Loaded strictly bounded {self.total_docs} semantic documents.")
        
        del df
        
        print("Mmapping embedding arrays for Phase 1...")
        self.embs = []
        for path in emb_paths:
            arr = np.load(path, mmap_mode='r')
            self.embs.append(arr)
            
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        self.batch_idx = 0
        
        self.indices = np.arange(self.total_docs)
        self.rng.shuffle(self.indices)
        self.num_batches = int(np.ceil(self.total_docs / self.batch_size))
        
    def __iter__(self):
        return self
        
    def __next__(self) -> Tuple[List[Any], Any]:
        if self.batch_idx >= self.num_batches:
            self.current_epoch += 1
            self.batch_idx = 0
            self.rng.shuffle(self.indices)
            raise StopIteration
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_docs)
        
        batch_doc_indices = self.indices[start_idx:end_idx]
        
        # 1. Fetch exactly the bounded chunks for each document
        raw_embs_list = [[] for _ in range(len(self.embs))]
        lengths = []
        
        for doc_idx in batch_doc_indices:
            start_pos = self.doc_start_indices[doc_idx]
            count = self.chunk_counts[doc_idx]
            
            if self.max_episode_len is not None:
                eff_len = min(count, self.max_episode_len)
            else:
                eff_len = min(count, 256) # Safety cap
                
            lengths.append(eff_len)
            
            for i in range(len(self.embs)):
                # Just copy the slice into memory
                raw_embs_list[i].append(np.array(self.embs[i][start_pos : start_pos+eff_len]))
                
        # 2. Pad to the length of the longest document in THIS specific batch (Dynamic Batching)
        max_len = max(lengths)
        
        # 2. Pad to the maximum document length in THIS specific batch
        padded_embs = [[] for _ in range(len(self.embs))]
        masks = []
        
        for b in range(len(batch_doc_indices)):
            l = lengths[b]
            pad_len = max_len - l
            
            for i in range(len(self.embs)):
                arr = raw_embs_list[i][b]
                embs_dim = arr.shape[-1]
                if pad_len > 0:
                    pad_array = np.zeros((pad_len, embs_dim), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad_array], axis=0)
                padded_embs[i].append(arr)
                
            masks.append([1] * l + [0] * pad_len)
            
        # Convert to arrays matching backend
        if self.backend == 'torch':
            import torch
            batch_embs_tensor = [torch.from_numpy(np.stack(padded_embs[i])).float() for i in range(len(self.embs))]
            masks_tensor = torch.tensor(masks, dtype=torch.float32)
            self.batch_idx += 1
            return batch_embs_tensor, masks_tensor
        else:
            import mlx.core as mx
            batch_embs_mx = [mx.array(np.stack(padded_embs[i])) for i in range(len(self.embs))]
            masks_mx = mx.array(masks)
            self.batch_idx += 1
            return batch_embs_mx, masks_mx
        
    def state_dict(self) -> Dict[str, Any]:
        return {"batch_idx": self.batch_idx, "current_epoch": self.current_epoch, "indices": self.indices.tolist()}
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.batch_idx = state["batch_idx"]
        self.current_epoch = state["current_epoch"]
        self.indices = np.array(state["indices"])
