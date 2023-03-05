import sqlalchemy as db
from sqlalchemy.engine import create_engine

def query_excute(query):
    # 데이터베이스 엔진 생성
    engine = create_engine('postgresql://postgres:qlrepdlxj@114.70.193.161/postgres')
    # 연결
    connection = engine.connect()
    # 쿼리문 받아서
    sql_query = query
    # 쿼리문 실행
    engine.execute(sql_query)