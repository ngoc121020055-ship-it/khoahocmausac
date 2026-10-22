import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Vá lỗi tương thích NumPy 2.0 cho colour-science
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }

for attr, target in [
    ('float', float),
    ('float_', np.float64),
    ('complex', complex),
    ('complex_', np.complex128),
    ('int', int),
    ('int_', np.int64),
    ('bool', bool),
    ('bool_', bool)
]:
    if not hasattr(np, attr):
        setattr(np, attr, target)

import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931
import streamlit as st
# Cấu hình trang
st.set_page_config(
    page_title="Khoa học Màu sắc - Ứng dụng Công nghệ In",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Ứng dụng Khoa học Màu sắc trong Công nghệ In")
st.markdown("*Chương trình phân tích không gian màu và đặc tính hóa máy in halftone*")

# Menu điều hướng ở thanh bên
option = st.sidebar.radio(
    "Danh mục nội dung:",
    [
        "1. Chuyển đổi Không gian Màu",
        "2. Tính giá trị 3 thành phần",
        "3. Biểu đồ CIE 1931 hình móng ngựa",
        "4. Gamut màu của thiết bị RGB",
        "5. Gamut màu của máy in Khoa",
        "6. Đặc tính hóa máy in nửa tone (Halftone)"
    ]
)
# ==============================================================================
# 1. CHUYỂN ĐỔI KHÔNG GIAN MÀU TỪ sRGB SANG XYZ VÀ NGƯỢC LẠI
# ==============================================================================
if option == "1. Chuyển đổi Không gian Màu" or option == "1. Chuyển đổi không gian màu":
    st.header("1. Chuyển đổi không gian màu (sRGB <-> XYZ)")
    
    direction = st.radio("Chọn chiều chuyển đổi:", ["sRGB -> XYZ", "XYZ -> sRGB"])
    
    if direction == "sRGB -> XYZ":
        st.subheader("Chuyển đổi từ sRGB (0 - 255) sang XYZ")
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.number_input("R (Red)", min_value=0, max_value=255, value=255)
        with col2:
            g = st.number_input("G (Green)", min_value=0, max_value=255, value=0)
        with col3:
            b = st.number_input("B (Blue)", min_value=0, max_value=255, value=0)
            
        srgb_vals = np.array([r, g, b]) / 255.0
        XYZ = colour.sRGB_to_XYZ(srgb_vals)
        
        st.success(f"**Kết quả XYZ:** X = `{XYZ[0]:.4f}`, Y = `{XYZ[1]:.4f}`, Z = `{XYZ[2]:.4f}`")
        
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(f'<div style="width:100%; height:50px; background-color:{hex_color}; border-radius:5px; margin-top:10px;"></div>', unsafe_allow_html=True)

    else:
        st.subheader("Chuyển đổi từ XYZ sang sRGB")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_val = st.number_input("X", min_value=0.0, max_value=1.0, value=0.4124)
        with col2:
            y_val = st.number_input("Y", min_value=0.0, max_value=1.0, value=0.2126)
        with col3:
            z_val = st.number_input("Z", min_value=0.0, max_value=1.0, value=0.0193)
            
        XYZ_vals = np.array([x_val, y_val, z_val])
        srgb_res = colour.XYZ_to_sRGB(XYZ_vals)
        srgb_clipped = np.clip(srgb_res, 0, 1)
        r_out, g_out, b_out = (srgb_clipped * 255).astype(int)
        
        st.success(f"**Kết quả sRGB (0-255):** R = `{r_out}`, G = `{g_out}`, B = `{b_out}`")
        
        hex_color = f"#{r_out:02x}{g_out:02x}{b_out:02x}"
        st.markdown(f'<div style="width:100%; height:50px; background-color:{hex_color}; border-radius:5px; margin-top:10px;"></div>', unsafe_allow_html=True)

# ==============================================================================
# 2. TÍNH GIÁ TRỊ 3 THÀNH PHẦN X, Y, Z TỪ FILE PHỔ ĐO
# ==============================================================================
elif option == "2. Tính giá trị 3 thành phần" or option == "2. Tính giá trị 3 thành phần X, Y, Z":
    st.header("2. Tính giá trị 3 thành phần X, Y, Z từ Phổ phản xạ")
    
    uploaded_file = st.file_uploader("Tải lên file dữ liệu phổ (.csv)", type=["csv"], key="sec2_file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Bảng dữ liệu phổ đầu vào:**")
        st.dataframe(df)
        
        try:
            # Lấy bước sóng và giá trị phổ phản xạ
            wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            reflectance = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
            
            if wavelengths.min() >= 300 and wavelengths.max() <= 830:
                spectral_dict = dict(zip(wavelengths, reflectance))
                sd = colour.SpectralDistribution(spectral_dict)
                
                # Tùy chọn nguồn sáng và quan sát viên
                col_a, col_b = st.columns(2)
                with col_a:
                    illuminant_name = st.selectbox("Chọn nguồn sáng (Illuminant):", ["D65", "A", "C", "D50", "F2"], index=0)
                with col_b:
                    observer_name = st.selectbox("Chọn Quan sát viên (Observer):", [
                        "CIE 1931 2 Degree Standard Observer",
                        "CIE 1964 10 Degree Standard Observer"
                    ], index=1)
                
                cmfs = colour.MSDS_CMFS[observer_name]
                illuminant = colour.SDS_ILLUMINANTS[illuminant_name]
                
                # Tính giá trị Tam kích thích XYZ
                XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant) / 100.0
                
                # Tính tọa độ màu x, y, z
                xy = colour.XYZ_to_xy(XYZ)
                
                st.subheader("Kết quả tính toán:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Giá trị X", f"{XYZ[0]:.4f}")
                c2.metric("Giá trị Y", f"{XYZ[1]:.4f}")
                c3.metric("Giá trị Z", f"{XYZ[2]:.4f}")
                
                c4, c5 = st.columns(2)
                c4.metric("Tọa độ x", f"{xy[0]:.4f}")
                c5.metric("Tọa độ y", f"{xy[1]:.4f}")
                
            else:
                st.warning("Cột đầu tiên của file phải chứa bước sóng phổ trong khoảng từ 380nm đến 780nm.")
        except Exception as e:
            st.error(f"Có lỗi khi xử lý file phổ: {e}")
# -----------------------------------------------------------------------------
# 3. BIỂU ĐỒ CIE HÌNH MÓNG NGỰA
# -----------------------------------------------------------------------------
if option == "3. Biểu đồ CIE 1931 hình móng ngựa":
    st.header("3. Biểu đồ sắc độ CIE 1931 (Horseshoe Curve)")
    fig, ax = plot_chromaticity_diagram_CIE1931(
        standalone=False, 
        show_spectral_locus=True, 
        title="Biểu đồ Sắc độ CIE 1931"
    )
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# 4. GAMUT MÀU CỦA THIẾT BỊ RGB
# -----------------------------------------------------------------------------
elif option == "4. Gamut màu của thiết bị RGB":
    st.header("4. Gamut màu của các thiết bị / chuẩn RGB")
    selected_spaces = st.multiselect(
        "Chọn không gian màu hiển thị:",
        ["sRGB", "Adobe RGB (1998)", "DCI-P3", "ProPhoto RGB"],
        default=["sRGB", "Adobe RGB (1998)"]
    )

    if selected_spaces:
        fig, ax = plot_chromaticity_diagram_CIE1931(standalone=False, title="So sánh Gamut RGB")
        colors = {'sRGB': 'blue', 'Adobe RGB (1998)': 'green', 'DCI-P3': 'purple', 'ProPhoto RGB': 'red'}

        for space in selected_spaces:
            rgb_space = colour.RGB_COLOURSPACES[space]
            primaries = rgb_space.primaries
            hull_pts = np.vstack([primaries, primaries[0]])
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], 'o-', label=space, color=colors.get(space, 'black'), linewidth=2)
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.15, color=colors.get(space, 'black'))

        ax.legend()
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# 5. GAMUT MÀU MÁY IN TỪ FILE GAMUT
# -----------------------------------------------------------------------------
elif option == "5. Gamut màu của máy in Khoa":
    st.header("5. Gamut màu của máy in (Dựa trên file đo)")
    uploaded_file = st.file_uploader("Tải lên file dữ liệu đo (.csv)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Đã tải dữ liệu thành công!")
        st.dataframe(df)

        pts = []

        # TRƯỜNG HỢP 1: File dữ liệu phổ (Spectral - Color_Cyan_Sample.csv)
        try:
            first_col_vals = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            if first_col_vals.min() >= 300 and first_col_vals.max() <= 830:
                spectral_dict = dict(zip(first_col_vals.dropna(), pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()))
                sd = colour.SpectralDistribution(spectral_dict)
                
                cmfs = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
                illuminant = colour.SDS_ILLUMINANTS['D65']
                XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant) / 100.0
                xy_point = colour.XYZ_to_xy(XYZ)
                
                fig, ax = plot_chromaticity_diagram_CIE1931(standalone=False, title="Tọa độ màu mẫu Cyan trên CIE 1931")
                ax.plot(xy_point[0], xy_point[1], 'ro', markersize=10, label=f"Mẫu Cyan (x={xy_point[0]:.4f}, y={xy_point[1]:.4f})")
                ax.legend()
                st.pyplot(fig)
        except Exception:
            pass

        # TRƯỜNG HỢP 2: File chứa cột x, y sẵn
        if 'x' in df.columns and 'y' in df.columns:
            pts = df[['x', 'y']].values

        # TRƯỜNG HỢP 3: File Lab (GamutMayIn_CMY_RGB_2.csv)
        elif 'ColorSpace' in df.columns or df.iloc[:, 0].astype(str).str.contains('L').any():
            color_cols = [c for c in df.columns if c not in ['ColorSpace', 'Unnamed: 0']]
            lab_data = []
            for col in color_cols:
                try:
                    l_val = df[df.iloc[:, 0] == 'L'][col].values[0]
                    a_val = df[df.iloc[:, 0] == 'a'][col].values[0]
                    b_val = df[df.iloc[:, 0] == 'b'][col].values[0]
                    lab_data.append([l_val, a_val, b_val])
                except Exception:
                    pass
            if lab_data:
                Lab = np.array(lab_data)
                XYZ = colour.Lab_to_XYZ(Lab)
                pts = colour.XYZ_to_xy(XYZ)

        # Vẽ Gamut cho Trường hợp 2 và 3
        if len(pts) >= 3:
            fig, ax = plot_chromaticity_diagram_CIE1931(standalone=False, title="Gamut Máy in trên CIE 1931")
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])

            ax.plot(hull_pts[:, 0], hull_pts[:, 1], 'k-o', label="Gamut Máy In Khoa", linewidth=2)
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.2, color='gray')
            ax.legend()
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# 6. ĐẶC TÍNH HÓA MÁY IN NỬA TONE (HALFTONE)
# -----------------------------------------------------------------------------
elif option == "6. Đặc tính hóa máy in nửa tone (Halftone)":
    st.header("6. Đặc tính hóa máy in nửa tone (Mô hình Murray-Davies)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nhập thông số mật độ quang")
        R_paper = st.number_input("Độ phản xạ của giấy (R_paper)", min_value=0.01, max_value=1.0, value=0.85, step=0.01)
        R_solid = st.number_input("Độ phản xạ mực nguyên chất (R_solid)", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
        
        a_nominal = np.linspace(0, 100, 11) / 100.0
        D_paper = -np.log10(R_paper)
        D_solid = -np.log10(R_solid)
        D_measured = D_paper + (D_solid - D_paper) * (a_nominal ** 0.75)
        
        a_effective = (1 - 10 ** -(D_measured - D_paper)) / (1 - 10 ** -(D_solid - D_paper))
        dot_gain = (a_effective - a_nominal) * 100

    with col2:
        st.subheader("Bảng tính gia tăng tầng thứ (TVI)")
        res_df = pd.DataFrame({
            "Tầng thứ thiết kế (%)": a_nominal * 100,
            "Tầng thứ thực tế (%)": np.round(a_effective * 100, 2),
            "Dot Gain (%)": np.round(dot_gain, 2)
        })
        st.dataframe(res_df)
        
    st.subheader("Đường cong gia tăng tầng thứ (TVI Curve)")
    fig_tvi, ax_tvi = plt.subplots(figsize=(8, 4))
    ax_tvi.plot(a_nominal * 100, a_effective * 100, 'b-o', label='Tầng thứ thực tế (Effective)')
    ax_tvi.plot(a_nominal * 100, a_nominal * 100, 'k--', label='Đường lý tưởng (Ideal)')
    ax_tvi.plot(a_nominal * 100, dot_gain, 'r--s', label='Mức gia tăng tầng thứ (Dot Gain)')
    ax_tvi.set_xlabel("Tầng thứ thiết kế (%)")
    ax_tvi.set_ylabel("Phần trăm (%)")
    ax_tvi.grid(True, linestyle=":")
    ax_tvi.legend()
    
    st.pyplot(fig_tvi)