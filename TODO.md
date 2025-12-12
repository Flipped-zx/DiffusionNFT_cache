# 实现过程
1.对于我现在的Container，在sample结束后所有的prompt各自拿一张（我在想要不要按r排序），同时在epoch结束后更新我的Container，同时我想存一个属于每一个prompt的标记 比如prompts_id等

2.在训练过程中，首先我容器中采样的K个一定要与我当前prompt是不一样的

3.对于我Batch内每个样本的计算而言，正只用我自己的+  负的r只用我自己的r+采样的K clean imgaes但是此时的r为0  即负loss部分是当前的$x^{-}$与$x_0（x_0\in K）$做MSE*1 所以对于一个样本来说 负loss是来自两部分的，正就只有他自己这一part

4.实现机制角度出发：我描述的这种方式如何做是更高效地