import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from step1 import generate_index_and_metadata

# 업로드된 파일 경로
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 결과 파일을 저장할 폴더
OUTPUT_FOLDER = './outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 예시로 사용할 파일 목록
documents = [
    "회의록 내용",
    "2025 상반기 대출실적 내용",
    "앱개발기획서 내용",
    "고객연락처 정리 내용",
    "투자보고서 2025Q1 내용"
]

# 임베딩 및 클러스터링, 인덱스 생성 - step1 호출
generate_index_and_metadata(documents, UPLOAD_FOLDER, OUTPUT_FOLDER)

# Streamlit - 웹에서 검색
os.system("streamlit run search-file.py")
