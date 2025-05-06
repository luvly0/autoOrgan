from step1 import generate_index_and_metadata

# 임베딩 및 클러스터링, 인덱스 생성 - step1 호출
generate_index_and_metadata()

# Streamlit - 웹에서 검색
os.system("streamlit run search-file.py")
