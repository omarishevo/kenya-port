import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="KPA Full Traffic Analysis", layout="wide")
st.title("📊 Kenya Ports Authority: Gate Traffic Data Summary")

# 1. Nationality Distribution
gender_data = {
    'Category': ['Truck Driver', 'Clearing Agents', 'Custom Officials', 'KPA Staffs', 'Traffic Police'],
    'Total': [714, 113, 10, 22, 9],
    '% Kenyans': [88.4, 96.5, 100, 100, 100],
    'Regional': [
        'Uganda (6.2%), Tanzania (2.2%),South Sudan (0.3%), Rwanda (1%), Burundi (0.8%), DRC Congo (1.1%)',
        'Uganda (0.9%), Tanzania (1.8%), Rwanda (0.9%)',
        'None', 'None', 'None']
}
st.header("1. 🌍 Nationality & Regional Representation")
st.dataframe(pd.DataFrame(gender_data))

# 2. Gender Category
gender_split = pd.DataFrame({
    'Category': ['Truck Drivers', 'Clearing Agents', 'Customs Officials', 'KPA Staff'],
    'Male': [700, 119, 7, 14],
    'Female': [14, 5, 3, 8],
    '% Male': [98.0, 96.0, 70.0, 63.6],
    '% Female': [2.0, 4.0, 30.0, 36.4]
})
st.header("2. 👤 Gender Distribution")
st.dataframe(gender_split)
fig, ax = plt.subplots()
gender_split.set_index('Category')[['Male', 'Female']].plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
plt.title("Gender Distribution by Stakeholder")
st.pyplot(fig)

# 3. Work Experience
data_exp = pd.DataFrame({
    'Work Experience': ['<1 year', '1-5 years', '6-10 years', '>10 years'],
    'Truck Driver': [43, 210, 133, 328],
    'Clearing Agents': [3, 2, 1, 4],
    'Custom Officials': [11, 50, 25, 38],
    'KPA Staff': [2, 1, 8, 11],
    'Traffic Police': [4, 5, 0, 0]
})
st.header("3. 💼 Work Experience")
st.dataframe(data_exp.set_index('Work Experience'))

# 4. Gate Visit Frequency
data_visits = pd.DataFrame({
    'Frequency': ['Daily', '2-3 times/week', 'Once a week', '1-3 times/month', '< Once a month'],
    'Truck Driver': [285, 176, 105, 138, 10],
    'Clearing Agents': [80, 35, 5, 4, 0],
    'Custom Official': [4, 3, 1, 1, 1],
    'Traffic Police': [9, 0, 0, 0, 0],
    'KPA Staff': [8, 14, 0, 0, 0]
})
st.header("4. 🚪 Gate Visit Frequency")
st.dataframe(data_visits.set_index('Frequency'))

# 5. Traffic Congestion Experience
data_congestion = pd.DataFrame({
    'Experience': ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    'Truck Drivers': [15, 48, 216, 190, 245],
    'Clearing Agents': [0, 6, 58, 39, 21]
})
st.header("5. 🚦 Traffic Congestion Experience")
st.dataframe(data_congestion.set_index('Experience'))

# 6. Waiting Time per Visit
data_wait = pd.DataFrame({
    'Time Category': ['<30 mins', '30 mins-1 hr', '1-2 hrs', '2-5 hrs', '>5 hrs'],
    'Truck Drivers': [87, 98, 119, 185, 245],
    'Clearing Agents': [27, 23, 26, 24, 24]
})
st.header("6. ⏱ Waiting Time at Gates")
st.dataframe(data_wait.set_index('Time Category'))

# 7. Gate Usage Distribution
data_gate = pd.DataFrame({
    'Gate': ['Gate 24', 'Gate 18', 'Main Gate 9/10', 'ICD', 'Others (12,13,15,16)'],
    'Truck Drivers': [475, 308, 93, 0, 0],
    'Clearing Agents': [74, 63, 27, 21, 0],
    'Customs Officials': [0, 0, 0, 10, 0],
    'Traffic Police': [0, 0, 0, 9, 0],
    'KPA Staff': [0, 0, 6, 12, 0]
})
st.header("7. 🛣 Gate Usage Distribution")
st.dataframe(data_gate.set_index('Gate'))

# 8. Congestion Time Frequency
data_time_congestion = pd.DataFrame({
    'Time': ['Morning', 'Midday', 'Afternoon', 'Evening'],
    'Truck Drivers': [150, 219, 480, 368],
    'Clearing Agents': [25, 22, 73, 76],
    'Custom Officials': [0, 0, 7, 3],
    'Traffic Police': [0, 0, 5, 5],
    'KPA Staff': [0, 5, 13, 12]
})
st.header("8. 🕐 Time of Day with Most Congestion")
st.dataframe(data_time_congestion.set_index('Time'))

# 9. Causes of Traffic Congestion
data_causes = pd.DataFrame({
    'Cause': [
        'Slow gate processing', 'Slow security checks', 'Documentation delays', 
        'KRA gadget delay', 'Poor road', 'Incomplete docs', 'Limited lanes'
    ],
    'Truck Drivers': [440, 377, 302, 316, 50, 40, 30],
    'Clearing Agents': [73, 49, 76, 69, 0, 0, 0],
    'Customs Officials': [0, 10, 10, 0, 0, 0, 0],
    'Traffic Police': [0, 12, 0, 9, 0, 0, 0],
    'KPA Staff': [0, 0, 10, 0, 0, 6, 10]
})
st.header("9. ❗ Causes of Traffic Congestion")
st.dataframe(data_causes.set_index('Cause'))

# 10. Effects on Work
data_effects = pd.DataFrame({
    'Effect': [
        'Longer working hours', 'Increased storage fees', 'Missed delivery schedules',
        'Increased fuel costs', 'Increased turnaround times', 'Stress/Fatigue', 'Demurrage charges', 'Cargo stacking delays'
    ],
    'Truck Drivers': [562, 386, 258, 305, 183, 0, 0, 0],
    'Clearing Agents': [84, 50, 57, 39, 0, 76, 35, 0],
    'Customs Officials': [0, 0, 0, 0, 0, 0, 0, 0],
    'KPA Staff': [0, 12, 0, 0, 0, 0, 10, 13]
})
st.header("10. 📉 Effects of Congestion on Work")
st.dataframe(data_effects.set_index('Effect'))
