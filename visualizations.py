import streamlit as st
import pandas as pd

st.set_page_config(page_title="KPA Full Traffic Analysis", layout="wide")
st.title("📊 Kenya Ports Authority: Gate Traffic Data Summary")

# 1. Nationality Distribution
st.header("1. 🌍 Nationality & Regional Representation")
nat_df = pd.DataFrame({
    'Category': ['Truck Driver', 'Clearing Agents', 'Custom Officials', 'KPA Staffs', 'Traffic Police'],
    'Total': [714, 113, 10, 22, 9],
})
nat_df["Percentage"] = nat_df["Total"] / nat_df["Total"].sum() * 100
st.dataframe(nat_df)
st.bar_chart(nat_df.set_index("Category")[["Total"]])

# 2. Gender Category
st.header("2. 👤 Gender Distribution")
gender_split = pd.DataFrame({
    'Category': ['Truck Drivers', 'Clearing Agents', 'Customs Officials', 'KPA Staff'],
    'Male': [700, 119, 7, 14],
    'Female': [14, 5, 3, 8]
})
gender_split['Total'] = gender_split['Male'] + gender_split['Female']
gender_split['% Male'] = (gender_split['Male'] / gender_split['Total']) * 100
st.dataframe(gender_split)
st.bar_chart(gender_split.set_index("Category")[['Male', 'Female']])

# 3. Work Experience
st.header("3. 💼 Work Experience")
data_exp = pd.DataFrame({
    'Work Experience': ['<1 year', '1-5 years', '6-10 years', '>10 years'],
    'Truck Driver': [43, 210, 133, 328]
})
data_exp["Total"] = data_exp["Truck Driver"]
data_exp["Percentage"] = data_exp["Truck Driver"] / data_exp["Truck Driver"].sum() * 100
st.dataframe(data_exp.set_index('Work Experience'))
st.bar_chart(data_exp.set_index('Work Experience')[['Truck Driver']])

# 4. Gate Visit Frequency
st.header("4. 🚪 Gate Visit Frequency")
data_visits = pd.DataFrame({
    'Frequency': ['Daily', '2-3 times/week', 'Once a week', '1-3 times/month', '< Once a month'],
    'Truck Driver': [285, 176, 105, 138, 10]
})
data_visits["Percentage"] = data_visits['Truck Driver'] / data_visits['Truck Driver'].sum() * 100
st.dataframe(data_visits.set_index('Frequency'))
st.bar_chart(data_visits.set_index('Frequency')[['Truck Driver']])

# 5. Traffic Congestion Experience
st.header("5. 🚦 Traffic Congestion Experience")
data_congestion = pd.DataFrame({
    'Experience': ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    'Truck Drivers': [15, 48, 216, 190, 245]
})
data_congestion["Percentage"] = data_congestion['Truck Drivers'] / data_congestion['Truck Drivers'].sum() * 100
st.dataframe(data_congestion.set_index('Experience'))
st.line_chart(data_congestion.set_index('Experience')[['Truck Drivers']])

# 6. Waiting Time per Visit
st.header("6. ⏱ Waiting Time at Gates")
data_wait = pd.DataFrame({
    'Time Category': ['<30 mins', '30 mins-1 hr', '1-2 hrs', '2-5 hrs', '>5 hrs'],
    'Truck Drivers': [87, 98, 119, 185, 245]
})
data_wait["Percentage"] = data_wait['Truck Drivers'] / data_wait['Truck Drivers'].sum() * 100
st.dataframe(data_wait.set_index('Time Category'))
st.line_chart(data_wait.set_index('Time Category')[['Truck Drivers']])

# 7. Gate Usage Distribution
st.header("7. 🛣 Gate Usage Distribution")
data_gate = pd.DataFrame({
    'Gate': ['Gate 24', 'Gate 18', 'Main Gate 9/10', 'ICD', 'Others (12,13,15,16)'],
    'Truck Drivers': [475, 308, 93, 0, 0]
})
data_gate["Percentage"] = data_gate['Truck Drivers'] / data_gate['Truck Drivers'].sum() * 100
st.dataframe(data_gate.set_index('Gate'))
st.bar_chart(data_gate.set_index('Gate')[['Truck Drivers']])

# 8. Congestion Time Frequency
st.header("8. 🕐 Time of Day with Most Congestion")
data_time_congestion = pd.DataFrame({
    'Time': ['Morning', 'Midday', 'Afternoon', 'Evening'],
    'Truck Drivers': [150, 219, 480, 368]
})
data_time_congestion["Percentage"] = data_time_congestion['Truck Drivers'] / data_time_congestion['Truck Drivers'].sum() * 100
st.dataframe(data_time_congestion.set_index('Time'))
st.bar_chart(data_time_congestion.set_index('Time')[['Truck Drivers']])

# 9. Causes of Traffic Congestion
st.header("9. ❗ Causes of Traffic Congestion")
data_causes = pd.DataFrame({
    'Cause': [
        'Slow gate processing', 'Slow security checks', 'Documentation delays', 
        'KRA gadget delay', 'Poor road', 'Incomplete docs', 'Limited lanes'
    ],
    'Truck Drivers': [440, 377, 302, 316, 50, 40, 30]
})
data_causes["Percentage"] = data_causes['Truck Drivers'] / data_causes['Truck Drivers'].sum() * 100
st.dataframe(data_causes.set_index('Cause'))
st.bar_chart(data_causes.set_index('Cause')[['Truck Drivers']])

# 10. Effects on Work
st.header("10. 📉 Effects of Congestion on Work")
data_effects = pd.DataFrame({
    'Effect': [
        'Longer working hours', 'Increased storage fees', 'Missed delivery schedules',
        'Increased fuel costs', 'Increased turnaround times'
    ],
    'Truck Drivers': [562, 386, 258, 305, 183]
})
data_effects["Percentage"] = data_effects['Truck Drivers'] / data_effects['Truck Drivers'].sum() * 100
st.dataframe(data_effects.set_index('Effect'))
st.bar_chart(data_effects.set_index('Effect')[['Truck Drivers']])
