# Processing Data


## 데이터셋 정보
- 프로젝트를 위한 dataset은 kaggle dataset에서 제공하는 정보를 사용하였음.
    - URL: https://www.kaggle.com/datafiniti/hotel-reviews
- Datafiniti의 비즈니스 DB에서 제공하는 1000개의 hotel review list이다.
- 데이터셋은 하나의 csv file로 주어지며, 약 40000개의 정보를 포함한다.
- 데이터셋은 호텔 이름, 호텔 위치, 작성자 이름, 평점, 리뷰 등의 정보를 포함한다.
    - 본 프로젝트는 평점(rating)과 리뷰(review text)만을 사용한다.
- 데이터셋의 리뷰 정보는 영어와 스페인어로 표기되어 있다. (인코딩: latin-1)


## 파일 설명
- data.csv: 제공하는 원본 dataset file
- data_rating_text.csv: data.csv에서 rating과 review text만 추출한 file
- data_final.csv: 데이터 정제 방법을 거친 후 최종적으로 사용하는 dataset file
- char2idx.pickle: 리뷰 기준에 대한 character 별로 숫자를 mapping한 python dictionary 객체 (pickle format)


## 데이터 정제 방법
- 평점 기준: 1 ~ 5 점의 정수형 데이터로 한정한다. [data에는 0 ~ 4점으로 표시되어 있다.]
    - 그 외의 평점(e.g. 0점, 실수 평점)은 폐기 및 반올림한다.
- 보장된 데이터 사용: 평점, 리뷰가 하나라도 존재하지 않은 정보는 사용하지 않는다.
- 리뷰 기준
    - 리뷰는 미리 정한 character만 취급한다. (그 외의 문자는 사전 삭제한다.)
    - 알파벳은 모두 소문자로 취급한다.
    - 미리 정한 character 기준 (46개): 숫자(0-9), 알파벳(a-z), 공백(ascii, unicode 모두 지원), 특정 특수문자(온점(.), 반점(,), 하이픈(-), 언더스코어(_), 느낌표(!), 물음표(?), 슬래시(/), 개행문자(\n))