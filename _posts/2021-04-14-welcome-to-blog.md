---
title: "CF기반 Binary Classifier 추천모델 만들어보기"
layout: post
---

[Kaggle에 올라와있는 데이터셋](https://www.kaggle.com/kandij/job-recommendation-datasets)을 이용해 구직자의 채용공고 페이지 방문이력을 바탕으로 특정 채용공고에 지원할지 안할지를 예측하는 Collaborative Filtering 기반 Binary Classifier 추천모델을 만들어본다.

## Dataset
총 5개의 데이터셋 중 구직자의 채용공고 페이지 방문이력 정보가 들어있는 `Job_Views.csv` 중 구직자 식별정보(`Applicant.ID`), 채용 식별정보(`Job.ID`) 및 페이지 확인일시(`View.Start`) 컬럼 만을 이용해볼 것이다.  
{% highlight ruby %}
train[['Applicant.ID','Job.ID','View.Start']].head(3)
{% endhighlight %}

| Applicant.ID          | Job.ID          | View.Start         |
|------------------|------------------|-----------------|
| 10000     | 73666     | 2014-12-12 20:12:35 UTC     |
| 10000	      | 96655      | 	2014-12-12 20:08:50 UTC   |
| 10001		      | 84141      | 	2014-12-12 20:12:32 UTC   |

## Data Preprocessing
이 실험에서는 페이지 확인일시(View.Start)값이 있는 경우 구직자의 관심있음을 1, 값이 없는 경우 0으로 하는 `checked` 컬럼을 새로 만들어 진행해볼 것이다. 크게 아래 순서로 진행한다.

1. 식별자 레이블링 작업
2. 페이지 확인일시(`View.Start`)를 1의 값을 가지는 새로운 컬럼 `checked`로 변환
3. 페이지 확인일시(`View.Start`)의 관계가 없는(즉 `checked`가 0인 경우) 구직자-채용정보 간 테이블을 생성하여 기존 테이블과 합침

위 1,2,3의 결과로 페이지 확인여부(`checked`)를 매개로 하는 구직자-채용정보 테이블을 만들 수 있으며 결과예시를 먼저 확인해보면 아래와 같은 형태가 되는 것을 목표로 한다.

| Applicant.ID          | Job.ID          | checked         |
|------------------|------------------|-----------------|
| 1     | 1     | 1     |
| 1	      | 2      | 	1   |
| 2		      | 3      | 	0   |



<!-- 
{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}
 -->