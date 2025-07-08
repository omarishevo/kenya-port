import streamlit as st
import pandas as pd
import plotly.express as px

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
nat_df = pd.DataFrame(gender_data)
st.dataframe(nat_df)
fig_nat = px.bar(nat_df, x='Category', y='Total', title='Stakeholder Count by Category')
st.plotly_chart(fig_nat)

# 2. Gender Category
gender_split = pd.DataFrame({
    'Category': ['Truck Drivers', 'Clearing Agents', 'Customs Officials', 'KPA Staff'],
    'Male': [700, 119, 7, 14],
    'Female': [14, 5, 3, 8]
})
st.header("2. 👤 Gender Distribution")
st.dataframe(gender_split)
fig_gender = px.bar(gender_split, x='Category', y=['Male', 'Female'], barmode='stack', title='Gender Breakdown')
st.plotly_chart(fig_gender)

# 3. Work Experience
st.header("3. 💼 Work Experience")
data_exp = pd.DataFrame({
    'Work Experience': ['<1 year', '1-5 years', '6-10 years', '>10 years'],
    'Truck Driver': [43, 210, 133, 328],
    'Clearing Agents': [3, 2, 1, 4],
    'Custom Officials': [11, 50, 25, 38],
    'KPA Staff': [2, 1, 8, 11],
    'Traffic Police': [4, 5, 0, 0]
})
st.dataframe(data_exp.set_index('Work Experience'))

# 4. Gate Visit Frequency
st.header("4. 🚪 Gate Visit Frequency")
data_visits = pd.DataFrame({
    'Frequency': ['Daily', '2-3 times/week', 'Once a week', '1-3 times/month', '< Once a month'],
    'Truck Driver': [285, 176, 105, 138, 10]
})
st.dataframe(data_visits.set_index('Frequency'))
fig_visits = px.bar(data_visits, x='Frequency', y='Truck Driver', title='Truck Driver Gate Visit Frequency')
st.plotly_chart(fig_visits)

# 5. Traffic Congestion Experience
st.header("5. 🚦 Traffic Congestion Experience")
data_congestion = pd.DataFrame({
    'Experience': ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    'Truck Drivers': [15, 48, 216, 190, 245]
})
st.dataframe(data_congestion.set_index('Experience'))
fig_congestion = px.pie(data_congestion, values='Truck Drivers', names='Experience', title='Truck Driver Congestion Experience')
st.plotly_chart(fig_congestion)

# 6. Waiting Time per Visit
st.header("6. ⏱ Waiting Time at Gates")
data_wait = pd.DataFrame({
    'Time Category': ['<30 mins', '30 mins-1 hr', '1-2 hrs', '2-5 hrs', '>5 hrs'],
    'Truck Drivers': [87, 98, 119, 185, 245]
})
st.dataframe(data_wait.set_index('Time Category'))
fig_wait = px.bar(data_wait, x='Time Category', y='Truck Drivers', title='Waiting Time Distribution')
st.plotly_chart(fig_wait)

# 7. Gate Usage Distribution
st.header("7. 🛣 Gate Usage Distribution")
data_gate = pd.DataFrame({
    'Gate': ['Gate 24', 'Gate 18', 'Main Gate 9/10', 'ICD', 'Others (12,13,15,16)'],
    'Truck Drivers': [475, 308, 93, 0, 0]
})
st.dataframe(data_gate.set_index('Gate'))
fig_gate = px.bar(data_gate, x='Gate', y='Truck Drivers', title='Gate Usage by Truck Drivers')
st.plotly_chart(fig_gate)

# 8. Congestion Time Frequency
st.header("8. 🕐 Time of Day with Most Congestion")
data_time_congestion = pd.DataFrame({
    'Time': ['Morning', 'Midday', 'Afternoon', 'Evening'],
    'Truck Drivers': [150, 219, 480, 368]
})
st.dataframe(data_time_congestion.set_index('Time'))
fig_time = px.bar(data_time_congestion, x='Time', y='Truck Drivers', title='Congestion Time Frequency')
st.plotly_chart(fig_time)

# 9. Causes of Traffic Congestion
st.header("9. ❗ Causes of Traffic Congestion")
data_causes = pd.DataFrame({
    'Cause': [
        'Slow gate processing', 'Slow security checks', 'Documentation delays', 
        'KRA gadget delay', 'Poor road', 'Incomplete docs', 'Limited lanes'
    ],
    'Truck Drivers': [440, 377, 302, 316, 50, 40, 30]
})
st.dataframe(data_causes.set_index('Cause'))
fig_causes = px.bar(data_causes, x='Cause', y='Truck Drivers', title='Causes of Congestion')
st.plotly_chart(fig_causes)

# 10. Effects on Work
st.header("10. 📉 Effects of Congestion on Work")
data_effects = pd.DataFrame({
    'Effect': [
        'Longer working hours', 'Increased storage fees', 'Missed delivery schedules',
        'Increased fuel costs', 'Increased turnaround times'
    ],
    'Truck Drivers': [562, 386, 258, 305, 183]
})
st.dataframe(data_effects.set_index('Effect'))
fig_effects = px.pie(data_effects, names='Effect', values='Truck Drivers', title='Effects of Congestion on Truck Drivers')
st.plotly_chart(fig_effects)
