import streamlit as st
import pandas as pd
import plotly.express as px
from Bio.PDB import PDBParser, Superimposer
from Bio import pairwise2
import os
import itertools
import py3Dmol
from stpyvmol import st_pyvmol
import numpy as np

# --- Helper Functions ---

def levenshtein_distance(s1, s2):
    """두 시퀀스 간의 편집 거리를 계산합니다."""
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_mutations(seq1, seq2):
    """두 시퀀스를 비교하여 변이 정보를 리스트로 반환합니다."""
    mutations = []
    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            mutations.append({"position": i + 1, "from": a, "to": b})
    return mutations

def calculate_rmsd(structure1, structure2):
    """두 PDB 구조를 중첩하고 RMSD 값을 계산합니다."""
    super_imposer = Superimposer()
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_name() == 'CA']
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_name() == 'CA']
    
    if len(atoms1) != len(atoms2):
        return None, "오류: 두 구조의 C-alpha 원자 수가 다릅니다."

    super_imposer.set_atoms(atoms1, atoms2)
    return super_imposer.rms, "성공"

def generate_llm_prompt(pair_data, rmsd, mutations):
    """분석 결과를 바탕으로 LLM 프롬프트를 생성합니다."""
    worse_binder, better_binder = pair_data['Worse_Binder'], pair_data['Better_Binder']
    worse_kd, better_kd = pair_data['Worse_KD'], pair_data['Better_KD']
    fold_improvement = pair_data['Fold_Improvement']
    
    mutation_str = ", ".join([f"{m['position']}번 위치의 {m['from']}를(을) {m['to']}(으)로 변경"] for m in mutations])
    
    prompt = f"""
### **분석 보고: Affinity Cliff 원인 분석 요청**

**1. 분석 대상:**
- **기준 바인더 (Worse):** {worse_binder} (KD: {worse_kd:.2f} nM)
- **개선 바인더 (Better):** {better_binder} (KD: {better_kd:.2f} nM)

**2. 핵심 변화 요약:**
- **활성도 변화:** 결합 친화도가 **{fold_improvement:.1f}배** 향상됨.
- **구조적 변이:** {mutation_str}.
- **구조적 유사도 (RMSD):** 두 구조의 전체적인 골격 차이는 **{rmsd:.3f} Å**으로 매우 유사함.

**3. 분석 요청:**
위 정보를 바탕으로, "{mutation_str}"라는 미세한 구조적 변이가 어떻게 결합 친화도를 {fold_improvement:.1f}배나 극적으로 향상시켰는지에 대한 **구조적, 물리화학적 가설 3가지를 제시해줘.** 각 가설은 PDB 구조 내에서 예상되는 **원자 간 상호작용(수소결합, 염다리, 소수성 상호작용 등)의 변화**를 구체적으로 언급해야 하며, 해당 가설을 검증하기 위한 **다음 실험 단계를 제안해줘.**
"""
    return prompt.strip()


# --- UI 설정 ---
st.set_page_config(layout="wide")
st.title("🔬 CAR-SAR: AI 바인더 분석 시스템 (v2.5)")
st.subheader("Phase 2.5: 심층 분석 및 가설 생성 준비")
st.write("---")

# --- 데모 데이터 생성 ---
def create_dummy_data_if_needed():
    """데모용 CSV와 PDB 파일이 없으면 생성합니다."""
    demo_csv_path = "data/binders_demo.csv"
    if not os.path.exists("data"): os.makedirs("data")
    if not os.path.exists(demo_csv_path):
        csv_data = """Binder_ID,Sequence (CDR3),Target_Antigen,KD (nM)
BKR-01,ARDYFGYGMDVW,BCMA,10.5
BKR-02,ARDYFWYGMDVW,BCMA,0.1
BKR-03,VRSKMDSSYFDY,BCMA,8.7
BKR-04,TRGSSYVLDAM,BCMA,120.3
BKR-05,GYDFWSGAYDY,BCMA,55.4
BKR-06,GYDFWSGAYEY,BCMA,2.1
BKR-07,ARDYFAYGMDVW,BCMA,9.8
"""
        with open(demo_csv_path, "w") as f: f.write(csv_data)
    pdb_dir = "data/pdb_files_demo"
    if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)
    pdb_content_template = "ATOM      1  N   ALA A   1      27.340  16.433  27.945  1.00  0.00           N\nATOM      2  CA  ALA A   1      26.266  15.548  27.433  1.00  0.00           C\n"
    binder_ids = ["BKR-01", "BKR-02", "BKR-03", "BKR-04", "BKR-05", "BKR-06", "BKR-07"]
    for binder_id in binder_ids:
        pdb_path = os.path.join(pdb_dir, f"{binder_id}.pdb")
        if not os.path.exists(pdb_path):
            with open(pdb_path, "w") as f: f.write(pdb_content_template)
    return demo_csv_path, pdb_dir

# --- 사이드바 및 데이터 로드 ---
st.sidebar.title("📁 데이터 소스")
default_csv_path, default_pdb_folder = create_dummy_data_if_needed()
csv_path = st.sidebar.text_input("1. 바인더 데이터 (CSV) 경로", value=default_csv_path)
pdb_folder_path = st.sidebar.text_input("2. PDB 파일 폴더 경로", value=default_pdb_folder)
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()

# --- Affinity Cliff 탐지 ---
st.header("1. Affinity Cliff 탐지")
col1, col2 = st.columns(2)
seq_diff_threshold = col1.number_input("서열 차이 (최대 편집 거리)", min_value=1, max_value=10, value=2)
kd_fold_threshold = col2.number_input("KD값 변화 (최소 배수)", min_value=2.0, max_value=1000.0, value=50.0, step=10.0)

if st.button("📈 Affinity Cliff 탐지 시작"):
    # ... (이전과 동일)
    cliff_pairs = []
    for (idx1, row1), (idx2, row2) in itertools.combinations(df.iterrows(), 2):
        seq1, seq2 = row1['Sequence (CDR3)'], row2['Sequence (CDR3)']
        dist = levenshtein_distance(seq1, seq2)
        if dist <= seq_diff_threshold:
            kd1, kd2 = row1['KD (nM)'], row2['KD (nM)']
            if kd1 > 0 and kd2 > 0:
                fold_change = kd1 / kd2 if kd1 > kd2 else kd2 / kd1
                if fold_change >= kd_fold_threshold:
                    better_binder, worse_binder = (row2, row1) if kd1 > kd2 else (row1, row2)
                    cliff_pairs.append({
                        "Pair": f"{worse_binder['Binder_ID']} vs {better_binder['Binder_ID']}",
                        "Worse_Binder": worse_binder['Binder_ID'], "Worse_Seq": worse_binder['Sequence (CDR3)'], "Worse_KD": worse_binder['KD (nM)'],
                        "Better_Binder": better_binder['Binder_ID'], "Better_Seq": better_binder['Sequence (CDR3)'], "Better_KD": better_binder['KD (nM)'],
                        "Fold_Improvement": fold_change, "Sequence_Distance": dist
                    })
    if cliff_pairs:
        st.success(f"총 {len(cliff_pairs)}개의 Affinity Cliff 쌍을 찾았습니다.")
        cliff_df = pd.DataFrame(cliff_pairs)
        st.dataframe(cliff_df[['Pair', 'Worse_KD', 'Better_KD', 'Fold_Improvement', 'Sequence_Distance']].style.format({"Fold_Improvement": "{:.1f}x", "Worse_KD": "{:.2f}", "Better_KD": "{:.2f}"}))
        st.session_state['cliff_df'] = cliff_df
    else:
        st.warning("설정된 기준으로 Affinity Cliff 쌍을 찾지 못했습니다.")

st.write("---")

# --- 심층 구조 비교 분석 ---
st.header("2. 🎯 심층 구조 비교 분석")

if 'cliff_df' in st.session_state:
    cliff_df = st.session_state['cliff_df']
    selected_pair_id = st.selectbox("분석할 Affinity Cliff 쌍을 선택하세요:", cliff_df['Pair'])
    
    if selected_pair_id:
        pair_data = cliff_df[cliff_df['Pair'] == selected_pair_id].iloc[0]
        
        # 데이터 추출
        worse_id, worse_seq, worse_pdb = pair_data['Worse_Binder'], pair_data['Worse_Seq'], os.path.join(pdb_folder_path, f"{pair_data['Worse_Binder']}.pdb")
        better_id, better_seq, better_pdb = pair_data['Better_Binder'], pair_data['Better_Seq'], os.path.join(pdb_folder_path, f"{pair_data['Better_Binder']}.pdb")

        if not os.path.exists(worse_pdb) or not os.path.exists(better_pdb):
            st.error("PDB 파일 중 하나 이상을 찾을 수 없습니다. 경로를 확인하세요.")
            st.stop()
            
        # PDB 파싱
        parser = PDBParser(QUIET=True)
        structure_worse = parser.get_structure("worse", worse_pdb)
        structure_better = parser.get_structure("better", better_pdb)
        
        # 분석 수행
        rmsd_value, rmsd_status = calculate_rmsd(structure_worse, structure_better)
        mutations = get_mutations(worse_seq, better_seq)

        # 분석 결과 표시
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("분석 요약")
            st.metric("구조적 유사도 (RMSD)", f"{rmsd_value:.3f} Å" if rmsd_value is not None else "계산 불가")
            if rmsd_status != "성공": st.warning(rmsd_status)
            
            st.text("서열 정렬 및 변이")
            # 간단한 텍스트 기반 서열 정렬 표시
            st.code(f"{worse_id}: {worse_seq}\n{better_id}: {better_seq}", language="text")
            
            mut_df = pd.DataFrame(mutations)
            st.table(mut_df)

        with col2:
            st.subheader("✍️ AI 가설 생성을 위한 프롬프트")
            llm_prompt = generate_llm_prompt(pair_data, rmsd_value, mutations)
            st.text_area("LLM 프롬프트", llm_prompt, height=300)

        # 3D 뷰어
        st.subheader("3D 구조 중첩 비교")
        view = py3Dmol.view(width=800, height=600)
        view.addModel(open(worse_pdb, 'r').read(), 'pdb', {'model': 0})
        view.addModel(open(better_pdb, 'r').read(), 'pdb', {'model': 1})
        view.superpose({'model': 1}, {'model': 0})
        
        # 기본 스타일
        view.setStyle({'model': 0}, {'cartoon': {'color': 'blue'}}) # Worse: Blue
        view.setStyle({'model': 1}, {'cartoon': {'color': 'red'}})  # Better: Red
        
        # 변이 잔기 하이라이트
        mut_positions = [m['position'] for m in mutations]
        if mut_positions:
            view.addStyle({'model': 0, 'resi': mut_positions}, {'stick': {'colorscheme': 'blueCarbon'}})
            view.addStyle({'model': 1, 'resi': mut_positions}, {'stick': {'colorscheme': 'redCarbon'}})
        
        view.zoomTo()
        st_pyvmol(view)

else:
    st.info("'Affinity Cliff 탐지 시작' 버튼을 눌러 분석을 먼저 실행해주세요.")

