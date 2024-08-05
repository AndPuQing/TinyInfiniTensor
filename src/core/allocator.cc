#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
        freeblock = std::map<size_t, size_t>();
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        if(!this->freeblock.empty()){
            for (auto it = this->freeblock.begin(); it != this->freeblock.end(); it++){
                //  ┌──────────────┐
                //  │    Tensor    │
                //  ├──────────────┤◄─────freeblock.first
                //  │              │  ▲
                //  │     Free     │  │   freeblock.second
                //  ├──────────────┤  ▼
                //  │              │
                //  │              │
                //  │    Tensor    │
                //  │              │
                //  │              │
                //  └──────────────┘
                if(it->second >= size){
                    size_t offset = it->first;
                    this->freeblock.erase(it);
                    this->freeblock[offset + size] = it->second - size;
                    this->used += size;
                    return offset;
                }
                //                         ┌──────────────┐
                //                         │    Tensor    │
                //                         ├──────────────┤
                //                         │              │
                //                         │    Tensor    │
                //   ┌─────────────┐       ├──────────────┤
                //   │             │       │              │
                //   │             │       │              │
                //   │             ├──────►│     Free     │
                //   │  NewTensor  │       │              │
                //   │             │       │              │
                //   │             │       └──────────────┴────►this.peak
                //   │             │
                //   │             │
                //   └─────────────┘
                if (it->first + it->second == this->peak){
                    size_t offset = it->first; 
                    this->used += size;
                    this->peak = std::max(this->peak, this->used);
                    this->freeblock.erase(it);
                    return offset;
                }
            }
        }
        size_t offset = this->used;
        this->used += size;
        this->peak = std::max(this->peak, this->used);
        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        this->freeblock[addr] = size;
        this->used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
