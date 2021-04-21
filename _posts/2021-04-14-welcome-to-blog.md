---
title: "Binary Classifier 추천모델 만들기"
layout: post
---

[Kaggle에 올라와있는 데이터셋](https://www.kaggle.com/kandij/job-recommendation-datasets)을 이용해 구직자의 채용공고 페이지 방문이력을 바탕으로 특정 채용공고에 지원할지 안할지를 예측하는 추천모델을 만들어본다.

## Dataset
총 5개의 데이터셋 중 구직자의 채용공고 페이지 방문이력 정보가 들어있는 `Job_Views.csv`만을 이용해볼 계획이다. 

{% highlight ruby %}
train[['Applicant.ID','Job.ID','View.Start']].head()
{% endhighlight %}

| Applicant.ID          | Job.ID          | View.Start         |
|------------------|------------------|-----------------|
| 10000     | 73666     | 2014-12-12 20:12:35 UTC     |
| 10000	      | 96655      | 	2014-12-12 20:08:50 UTC   |
| 10001		      | 84141      | 	2014-12-12 20:12:32 UTC   |
| 10002	      | 77989      | 	2014-12-12 20:39:23 UTC   |
| 10002	      | 69568      | 	2014-12-12 20:43:25 UTC   |

<!-- 
{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}
 -->