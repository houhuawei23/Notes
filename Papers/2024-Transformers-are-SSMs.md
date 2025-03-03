Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
Transformer 是 SSM：通过结构化状态空间对偶性实现的通用模型和高效算法
Tri Dao, Albert Gu
杜里道，阿尔伯特·古

[text](https://arxiv.org/abs/2405.21060)

While Transformers have been the main architecture behind deep learning's success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale. We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices. Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba's selective SSM that is 2-8X faster, while continuing to be competitive with Transformers on language modeling.
虽然 Transformer 一直是深度学习在语言建模方面成功的主要架构，但最近研究表明，如 Mamba 这样的状态空间模型（SSM）在小到中等规模上可以匹配或超越 Transformer。我们表明，这些模型家族实际上非常密切相关，并开发了一个丰富的理论框架，将 SSM 与注意力变体之间的理论联系联系起来，这些联系通过一系列结构半可分离矩阵的分解实现。我们的状态空间对偶（SSD）框架使我们能够设计一个新的架构（Mamba-2），其核心层是对 Mamba 选择性 SSM 的改进，速度提高了 2-8 倍，同时在语言建模方面继续与 Transformer 保持竞争力。