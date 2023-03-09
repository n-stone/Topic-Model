# Topic Modeling Processes

## 1. LDA(Latent Dirichlet Allocation)
- ### LDA는 사용자가 토픽 수를 설정해야 한다.
- ### LDA는 문서에 대한 토픽 분포와 각 토픽에 속하는 단어 분포를 추정하는 단일 계층 모델이다.
- ### LDA는 Gibbs 샘플링 등의 메트로폴리스-헤이스팅스 알고리즘을 사용하여 추론을 수행한다.
- ### LDA는 HDP보다 덜 복잡한 구조로 계산비용이 작기 때문에 수행속도는 빠르다.
<br>

## 2. HDP(Hierarchical Dirichlet Processes)
- ### HDP는 토픽 수를 자동으로 학습한다.
- ### HDP는 토픽 간에 계층 구조를 가질 수 있으며, 이를 통해 더 복잡한 데이터 구조를 모델링할 수 있다.
- ### HDP는 양자화를 사용하여 추론을 수행한다.
- ### HDP는 LDA보다 복잡한 구조로 계산비용이 크기 때문에 수행속도는 느리다.
<br>

## 3. Referece
- [LDA(Latent Dirichlet Allocation)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [HDP(Hierarchical Dirichlet Processes)](https://people.eecs.berkeley.edu/~jordan/papers/hdp.pdf)
