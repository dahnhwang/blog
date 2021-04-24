---
title: "CF기반 Binary Classifier 추천모델 만들어보기"
layout: post
---

[Kaggle에 올라와있는 데이터셋](https://www.kaggle.com/kandij/job-recommendation-datasets)을 이용해 구직자의 채용공고 페이지 방문이력을 바탕으로 특정 채용공고에 지원할지 안할지를 예측하는 Collaborative Filtering 기반 Binary Classifier 추천모델을 만들어본다.



## Dataset
총 5개의 데이터셋 중 구직자의 채용공고 페이지 방문이력 정보가 들어있는 `Job_Views.csv` 중 구직자 식별정보(`Applicant.ID`), 채용 식별정보(`Job.ID`) 및 페이지 확인일시(`View.Start`) 컬럼 만을 이용해볼 것이다.  
{% highlight python %}
raw[['Applicant.ID','Job.ID','View.Start']].head(3)
{% endhighlight %}

| Applicant.ID          | Job.ID          | View.Start         |
|------------------|------------------|-----------------|
| 10000     | 73666     | 2014-12-12 20:12:35 UTC     |
| 10000	      | 96655      | 	2014-12-12 20:08:50 UTC   |
| 10001		      | 84141      | 	2014-12-12 20:12:32 UTC   |

{% highlight python %}
user_count = raw['Applicant.ID'].unique().shape[0]
job_count = raw['Job.ID'].unique().shape[0]
print(user_count, job_count) 
{% endhighlight %}

3448명의 구직자와 7047개의 채용공고로 이루어진 테이블이다. 



## Data Preprocessing Planning
이 실험에서는 페이지 확인일시(View.Start)값이 있는 경우 구직자의 관심있음을 뜻하는 1, 값이 없는 경우 0으로 하는 `checked` 컬럼을 새로 만들어 진행해볼 것이다. 크게 아래 순서로 진행한다.

1. 식별자 레이블링 작업
2. 페이지 확인일시(`View.Start`)를 1의 값을 가지는 새로운 컬럼 `checked`로 변환
3. 페이지 확인일시(`View.Start`)의 관계가 없는(즉 `checked`가 0인 경우) 구직자-채용정보 간 테이블을 생성하여 기존 테이블과 합침

위 1,2,3의 결과로 페이지 확인여부(`checked`)를 매개로 하는 구직자-채용정보 테이블을 만들 수 있으며 아래와 같은 결과예시의 데이터를 모델학습을 위한 Input으로 사용할 것이다.

| Applicant.ID          | Job.ID          | checked         |
|------------------|------------------|-----------------|
| 1     | 1     | 1     |
| 1	      | 2      | 	1   |
| 2		      | 3      | 	0   |



## Data Preprocessing


### 1. 식별자 레이블링 작업

{% highlight python %}
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
raw['Applicant.ID'] = user_enc.fit_transform(raw['Applicant.ID'].values)

item_enc = LabelEncoder()
raw['Job.ID'] = item_enc.fit_transform(raw['Job.ID'].values)
{% endhighlight %}

sklearn의 LabelEncoder를 사용하여 10000부터 매겨져있는 기존 테이블의 ID를 정수 1부터 시작하는 것으로 새롭게 라벨링을 했다. 데이터셋에서 사용자의 ID가 정수로 되어있는 경우는 드물기 때문에 무조건 처리해주고 시작하는게 좋은 것 같다.


### 2. 페이지 확인일시(`View.Start`)를 1의 값을 가지는 새로운 컬럼 `checked`로 변환

{% highlight python %}
count = raw['View.Start'].isnull().sum()
count 
{% endhighlight %}

페이지 확인일시(`View.Start`) 정보가 없는 행이 하나도 없기 때문에 이를 새로운 컬럼으로 변환하는 작업을 계속 진행한다.

{% highlight python %}
raw['checked'] = 1
raw = raw.drop(['View.Start'], axis=1)
raw.head(3)
{% endhighlight %}

| Applicant.ID          | Job.ID          | checked         |
|------------------|------------------|-----------------|
| 980     | 1610     | 1     |
| 980	      | 1610      | 	1   |
| 981		      | 2989      | 	1   |

그런데 위 테이블에서 볼 수 있듯이 한 명의 구직자가 동일한 채용공고를 여러번 확인했을 수 있으므로 중복되는 행은 제외하도록 한다.

{% highlight python %}
train.drop_duplicates(subset=None, keep="first", inplace=True)
{% endhighlight %}

이렇게 하여 `checked`의 값이 1인 총 8920행으로 된 테이블이 되었다.


### 3. 페이지 확인일시(`View.Start`)의 관계가 없는(즉 `checked`가 0인 경우) 구직자-채용정보 간 테이블을 생성하여 기존 테이블과 합침

주어진 데이터셋의 경우 구직자가 채용공고를 확인한 데이터만 존재하고 그렇지 않은 경우의 데이터는 존재하지 않는다. 따라서 확인하지 않은 데이터를 새로 생성하기로 하였다.

{% highlight python %}
user_uid = raw['Applicant.ID'].unique()
job_uid = raw['Job.ID'].unique()
user_uid_table = DataFrame({'Applicant.ID':user_uid})
job_uid_table = DataFrame({'Job.ID':job_uid})
{% endhighlight %}

구직자 및 채용공고의 unique 값으로 이루어진 각각의 다른 두 개의 테이블 `user_uid_table` 와 `job_uid_table`을 생성해 이 둘을 곱집합(Cartesian product)한 테이블을 만들 것이다. 즉, 모든 구직자에 대한 모든 채용공고의 관계(`full_table`)를 나타낼 것이다. 

{% highlight python %}
user_uid_table['checked'] = 0
job_uid_table['checked'] = 0
{% highlight python %}

{% highlight python %}
user_uid_table.head(3)
{% endhighlight %}

| Applicant.ID          | checked          |
|------------------|------------------|
| 980     | 0     |
| 981	      | 0      |
| 982		      | 0      |

{% highlight python %}
job_uid_table.head(3)
{% endhighlight %}

| Job.ID          | checked          |
|------------------|------------------|
| 1610     | 0     |
| 1611	      | 0      |
| 1612		      | 0      |

위와 같은 3448*2짜리 `user_uid_table`과 7047*2짜리 `job_uid_table`을 `checked`를 통해 조인시켜 `full_table`을 만든다.

{% highlight python %}
full_table = user_uid_table.merge(job_uid_table, on='checked') # 24298056 
{% endhighlight %}

<!-- 결과로 3,448*7,047의 결과값인 24,298,056개의 행을 가진 테이블이 생성되었다.

이제 기존의 `raw` 테이블과 `full_table`을 concat을 통해 합친다음, 중복되는 데이터 중 `checked`값이 0인 경우는 삭제하여 모델에 입력할 최종 데이터셋을 만들 것이다. -->
<!-- 
{% highlight python %}
concat_df = pd.concat([full_table, raw]) 
concat_df
{% endhighlight %} -->

<!-- 중복제거 전 테이블의 행끼리 그냥 합친 `concat_df`는 24306976 rows × 3 columns로 확인된다.

{% highlight python %}
train = concat_df.drop_duplicates(subset=['Applicant.ID', 'Job.ID'], keep='last')
{% endhighlight %}

 -->