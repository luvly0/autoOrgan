import streamlit as st
from step1 import generate_index_and_metadata

# 1. 인덱스 및 메타데이터 생성 (처음 한 번만 실행되도록 조건 처리도 가능)
if st.button("인덱스/메타데이터 생성하기"):
    generate_index_and_metadata()
    st.success("임베딩, 클러스터링, 인덱스 생성 완료!")

# 2. 검색 입력
st.markdown("## 🔍 유사한 문서 검색")
query = st.text_input("검색어를 입력하세요:")

if query:
    # 여기서 search-file.py 안의 검색 함수를 불러오도록 처리
    from search_file import search_similar_files
    results = search_similar_files(query)

    st.write("### 📄 검색 결과:")
    for item in results:
        st.write(f"- {item}")
