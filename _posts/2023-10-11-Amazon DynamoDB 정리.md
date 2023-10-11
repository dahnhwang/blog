---
title: "Amazon DynamoDB 정리"
layout: post
---

!https://changhoi.kim/images/2022-04-16-dynamodb-internals-1/thumbnail.png

## DynamoDB란?

- RDS :  AWS에서 제공하는 관계형 데이터베이스 서비스
- DynamoDB : NoSQL 데이터베이스. 10밀리초 미만의 응답시간을 제공하고 더 빠른 액세스가 필요한 경우 Dynamo DB Accelerator 메모리 캐시를 제공

## NoSQL이란?

- Not Only SQL
- 데이터 모델을 어떻게 설계하느냐에 따라 데이터베이스의 포맷이 달라짐

| 문서 데이터베이스 | 주로 JSON형태의 문서 보관 |
| --- | --- |
| Key-Value DB | Key-Value 형태의 JSON데이터 보관 |
| 그래프 데이터베이스 | 노드와 엣지를 이용해서 데이터를 보관 |

## DynamoDB 장점*

- 빠른 속도 : 테이블 비정규화(중복데이터 포함으로 다른 테이블과 합치는 join 연산이 필요없음)로 쓰기 퍼포먼스는 낮지만 읽기 퍼포먼스가 매우 높음
- 오토스케일링 기능 : 데이터의 처리량에 따라 성능을 늘렸다 줄였다
- 테이블 생성 시 기본키를 제외한 어떤 것도 미리 정의할 필요 없음. (스키마 등 정의 필요 없음)

## 기본키

DynamoDB에서는 두 가지 기본키를 지원. 파티션키와 복합키. 둘 중 하나를 반드시 선택해야 함.

1. 파티션키 : DynamoDB 내부에 있는 물리적 스토리지를 파티션이라 한다.
    
                    파티션키는 테이블에 있는 데이터를 파티션으로 나누고 분리시키는데 사용하는 키이다.
    
                    주로 높은 카디널리티(중복이 별로 없는 것. ex 회원ID)가 파티션 키로 주로 사용됨.
    
2. 복합키 : 위의 파티션키와 정렬키(sort key)를 합쳐놓은 것.

                    정렬키는 데이터가 정렬될 때 사용되는 기준점이며 ‘구매일자’같이 날짜 타입의 열이 주로 선택됨.

## DynamoDB의 인덱스

쿼리 성능을 더 끌어올릴 수있는 방법. 테이블 전체를 스캔하는 것이 아닌 특정 열을 기준으로 쿼리 진행.

- 로컬 보조 인덱스 (Local Secondary Index) : 복합키(파티션키+정렬키) 사용. 정렬키를 사용해 정렬된 데이터에서 하나의 파티션키가 들어있는 테이블을 주로 쿼리하는 상황이 발생할 경우 권장됨. 테이블 생성 시 변경삭제 불가.
- 글로벌 보조 인덱스 (Global Secondary Index) : 정렬키는 선택사항. 테이블 생성 후에도 추가변경삭제 가능.

## DynamoDB에서 데이터를 가져오는 방법 두 가지

- 쿼리 : 정의한 기본키를 가지고 데이터를 가져오는 방법.
- 스캔 : 기본키를 사용하지 않고 테이블 안에 있는 모든 데이터를 불러오는 방법. 하지만 한번에 다 가져오는 것은 아니고 1MB씩에 해당하는 배치 데이터를 반환. (테이블 크기가 100MB라면 총 100번의 배치 데이터 반환) 스캔 성능향상을 위해 병렬 스캔을 통해 데이터를 가져오기도 함.
- 실시간으로 데이터가 자주 업데이트되지 않고 데이터 중복이 없는 경우 스캔을 사용해도 무방.

## **읽기/쓰기 용량 모드**

Amazon DynamoDB에는 테이블에서 읽기 및 쓰기 처리를 위한 두 가지 읽기/쓰기 용량 모드가 존재.

- 온디맨드 모드 : 용량 계획 없이 초당 수천 개의 요청을 처리할 수 있는 유연한 청구 옵션. 사용하는 만큼에 대해서만 비용을 지불
    - 알 수 없는 워크로드를 포함하는 테이블을 새로 만들 경우
    - 애플리케이션 트래픽이 예측 불가능한 경우
    - 사용한 만큼에 대해서만 지불하는 요금제를 사용하려는 경우
- 프로비저닝 모드 : 기본값, 프리 티어 이용 가능. 개발자가 애플리케이션에 필요한 초당 읽기 및 쓰기 횟수를 지정하고 그만큼의 비용을 지불. 오토스케일링을 사용하여 지정된 사용률을 기준으로 테이블의 용량을 자동으로 조정하여 비용을 절감
    - 애플리케이션 트래픽이 예측 가능한 경우
    - 트래픽이 일관되거나 점진적으로 변화하는 애플리케이션을 실행할 경우
    - 비용 관리를 위해 용량 요구 사항을 예측할 수 있는 경우

## Amazon DynamoDB Accelerator(DAX)

[Amazon DynamoDB](https://aws.amazon.com/ko/dynamodb/)를 위해 구축된 고가용성의 완전관리형 캐싱 서비스. 

- DynamoDB에서 일관되게 10밀리초 미만의 지연 시간을 제공하지만, DynamoDB와 DAX가 결합되면 성능을 한 단계 업그레이드하여 읽기 중심의 워크로드에서 초당 수백만 개의 요청에도 마이크로초의 응답 시간을 지원

!https://d1.awsstatic.com/product-marketing/DynamoDB/dax_high_level.e4af7cc27485497eff5699cdf22a9502496cba38.png

## TEST

솔루션 설계자는 트래픽이 많은 전자 상거래 웹 애플리케이션을위한 데이터베이스 솔루션을 설계해야합니다. 데이터베이스는 고객 프로필과 장바구니 정보를 저장합니다. 데이터베이스는 초당 수백만 요청의 최대로드를 지원하고 밀리 초 내에 응답을 전달해야합니다. 데이터베이스 관리 및 확장을 위한 운영 오버 헤드를 최소화해야합니다. 솔루션 설계자가 권장해야하는 데이터베이스 솔루션은 무엇입니까?

A. Amazon Aurora

B. Amazon DynamoDB

C. Amazon RDS

D. Amazon Redshift

---

A company deployed a serverless application that uses Amazon DynamoDB as a database layer. The application has experienced a large increase in users. The company wants to improve database response time from milliseconds to microseconds and to cache requests to the database.

Which solution will meet these requirements with the LEAST operational overhead?

- A. Use DynamoDB Accelerator (DAX).
- B. Migrate the database to Amazon Redshift.
- C. Migrate the database to Amazon RDS.
- D. Use Amazon ElastiCache for Redis.


<font color='#909194'>Last updated: October 11, 2023</font>