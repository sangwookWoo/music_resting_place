### KoELECTRA 를 사용한 pretrained 모델을 만들며 느낌 어려운점.

![모델의 기존 input 형태](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cf549924-2db4-493e-97f6-2e1fcc47b511/Untitled.png)

모델의 기존 input 형태

1. 위 사진처럼 Embedding 형식이 이미 정해져 있고, 그렇기 때문에 입력형태를 원하는 방식으로 바꿀 경우 이미 학습시킨 모델에 악영향을 미칠것으로 보였음.

![모델의 기존 output 형태](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/119d518f-9ffe-41be-ac9a-4aafaacc335a/Untitled.png)

모델의 기존 output 형태

2. 그래서 우리의 데이터를 동일한 입력형태로 변경하여 모델에 넣은 후 위 사진처럼 형성되어 있는 마지막 output층만을 변경하여 좋다,싫다 2중 분류를 하는 모델에서 5가지 감정을 분류하는 다중 분류 모델을 만들려했으나(아래 사진 참고) 애초에 2중분류로 설계되어 있는 모델에 출력형태만 억지로 변경했다 보니 data type, input data size 에러가 발생했음.(해결시에 너무 오랜시간이 걸릴것으로 예상되었기에 프로젝트에 악영향을 미칠 것으로 예상됨)

![변경한 output 형태](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d2d1b43-0efb-40bb-9531-dde0932ef050/Untitled.png)

변경한 output 형태

3. 또한 현재까지 개발한 catboost의 예측 성능이 0.69인 것을 감안하여 DL pretrained 모델을 완성시키는 것보다 우리의 모델 성능을 올리는 쪽이 더 합리적이라는 판단이 들었음. 
4. colab의 GPU RAM이 모자란 에러가 자주 발생했음. batch_size를 손보기도 했으나 무료 코랩시간으로 학습자체가 어려울 것으로 보임.
5. [https://huggingface.co/jaehyeong/koelectra-base-v3-generalized-sentiment-analysis](https://huggingface.co/jaehyeong/koelectra-base-v3-generalized-sentiment-analysis) 에 들어가서 aws studio lab의 허용을 받거나 나중에 aws 인증받는 방법을 알아내서 이 코드를 돌려볼수도 있을듯
6. torch pretrained model 쉽게 사용하는 법 블로그 [https://yeko90.tistory.com/entry/pytorch-pretrained-model](https://yeko90.tistory.com/entry/pytorch-pretrained-model)
7. 참조한 코드 [https://heegyukim.medium.com/huggingface-koelectra로-nsmc-감성분류모델학습하기-1a23a0c704af](https://heegyukim.medium.com/huggingface-koelectra%EB%A1%9C-nsmc-%EA%B0%90%EC%84%B1%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-1a23a0c704af)