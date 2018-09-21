

# Data Pipelines

- 6 July 2017
- UWCSE Deep Learning in Practice Summer 2017
- Presenter: John 
- Joseph Zhong

## Data Pipelines

- SATA3: 600MB/s
- NVMe: ~1GB/s
- 40 PCIe Lanes: 8x 1GB/s

### Storing the Dataset

- 1GB: Store it on GPU
  - Fastest option
- 128GB: Store it on RAM
- 512GB: Still store on RAM


### First-Order Method Template

```
w^{(1)}
for t=1... do
  Lmb = Random Minibatch
  w = First order update w^{(t-1)} w.r.t. Lmb
end
```

### Random Mini-Batch

For ImageNet as an example:

- Minibatch is ~75MB
- ~8 minibatches/s
- Multiple concurrent experiments
- Minibatch rate drops linearly


## Multiple Concurrent Experiments

Data Server

- Main Thread
  - Reads disk
  - Fills Queues
- Client Threads
  - Serve data to client from queue over socket

Data Client
  - Get batch from socket



### Concurrent Experiments Pt. 2

- Linux Pagebuffer
  - https://www.thomas-krenn.com/en/wiki/Linux_Page_Cache_Basics
  - Simply let the Linux Pagebuffer handle the same process as previous in cache
  - Each Data Client read from Disk and eventually the dataset becomes cached
- IF you can fit complete dataset into RAM
- Monitor with `free -m`
  - `buff/cache` tracks the size of the pagebuffer
- Clear the Page buffer for debugging
  - `sudo sh -c 'echo 1 >/proc/sys/vm/drop_caches'`

#### Alternatives:

- `VM Touch`: Lock
- `numpy`: Memory Map
  - https://docs.scipy.org/doc/numpy-1.11.0/reference/generated/numpy.memmap.html

### Final Thoughts

- Seek times are not O(1) due to the filesystem
  - SSDs advertise O(1) seek time but filesystems store in fragments
- Consider the following:
  - Storing data on an empty partition
  - HDF5 Database: Faster than filesystem seeking
  - Store multiple chunks
  - Multiple SSDs with RAID
    - RAID0 multiplies bandwidth by factor of number of drives
      - Parallel reads
    - RAID SSD or NVMe is the probably the best option

- To kill all your ram for experiments

```
memory_eater = ''*56*2**30
```



