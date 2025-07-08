import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="KPA Full Traffic Analysis", layout="wide")

# Custom background
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Kenya Ports Authority: Gate Traffic Data Summary")

def pie_chart(df, value_field, category_field, title):
    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=value_field,
        color=category_field,
        tooltip=[category_field, value_field, 'Percentage']
    ).properties(title=title)

    text = alt.Chart(df).mark_text(radius=100, size=12, color='black').encode(
        theta=alt.Theta(value_field, stack=True),
        text=alt.Text('Percentage:Q', format='.1f')
    )
    return chart + text

# 1. Nationality Distribution
st.header("1. 🌍 Nationality & Regional Representation")
nat_df = pd.DataFrame({
    'Category': ['Truck Driver', 'Clearing Agents', 'Custom Officials', 'KPA Staffs', 'Traffic Police'],
    'Total': [714, 113, 10, 22, 9],
})
nat_df["Percentage"] = nat_df["Total"] / nat_df["Total"].sum() * 100
st.dataframe(nat_df)

fig1 = alt.Chart(nat_df).mark_bar().encode(
    x=alt.X('Category', sort='-y'),
    y='Total',
    color='Category',
    tooltip=['Category', 'Total', 'Percentage']
).properties(title="Total Stakeholders by Category")

labels1 = alt.Chart(nat_df).mark_text(
    dy=-10,
    size=13,
    fontWeight='bold',
    color='black',
    stroke='white',
    strokeWidth=2
).encode(
    x='Category',
    y='Total',
    text=alt.Text('Percentage:Q', format='.1f')
)

st.altair_chart(fig1 + labels1, use_container_width=True)

# 2. Gender Category
st.header("2. 👤 Gender Distribution")
gender_split = pd.DataFrame({
    'Category': ['Truck Drivers', 'Clearing Agents', 'Customs Officials', 'KPA Staff'],
    'Male': [700, 119, 7, 14],
    'Female': [14, 5, 3, 8]
})
gender_split['Total'] = gender_split['Male'] + gender_split['Female']
gender_split['% Male'] = (gender_split['Male'] / gender_split['Total']) * 100
gender_split['% Female'] = 100 - gender_split['% Male']
st.dataframe(gender_split)

long_gender = gender_split.melt(id_vars='Category', value_vars=['Male', 'Female'],
                                var_name='Gender', value_name='Count')
long_gender["Percentage"] = long_gender.groupby('Category')['Count'].transform(
    lambda x: 100 * x / x.sum())

fig2 = alt.Chart(long_gender).mark_bar().encode(
    x='Category',
    y='Count',
    color='Gender',
    tooltip=['Category', 'Gender', 'Count', 'Percentage']
)

labels2 = alt.Chart(long_gender).mark_text(
    dy=-10,
    size=13,
    fontWeight='bold',
    color='black',
    stroke='white',
    strokeWidth=2
).encode(
    x='Category',
    y='Count',
    text=alt.Text('Percentage:Q', format='.1f')
)

st.altair_chart(fig2 + labels2, use_container_width=True)

# 3. Work Experience
st.header("3. 💼 Work Experience")
data_exp = pd.DataFrame({
    'Work Experience': ['<1 year', '1-5 years', '6-10 years', '>10 years'],
    'Truck Driver': [43, 210, 133, 328]
})
data_exp["Percentage"] = data_exp["Truck Driver"] / data_exp["Truck Driver"].sum() * 100
st.dataframe(data_exp)
st.altair_chart(pie_chart(data_exp, 'Truck Driver', 'Work Experience', "Truck Driver Work Experience (Pie Chart)"))

# 4. Gate Visit Frequency
st.header("4. 🚪 Gate Visit Frequency")
data_visits = pd.DataFrame({
    'Frequency': ['Daily', '2-3 times/week', 'Once a week', '1-3 times/month', '< Once a month'],
    'Truck Driver': [285, 176, 105, 138, 10]
})
data_visits["Percentage"] = data_visits['Truck Driver'] / data_visits['Truck Driver'].sum() * 100
st.dataframe(data_visits)
st.altair_chart(pie_chart(data_visits, 'Truck Driver', 'Frequency', "Gate Visit Frequency (Pie Chart)"))

# 5. Traffic Congestion Experience
st.header("5. 🚦 Traffic Congestion Experience")
data_congestion = pd.DataFrame({
    'Experience': ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    'Truck Drivers': [15, 48, 216, 190, 245]
})
data_congestion["Percentage"] = data_congestion['Truck Drivers'] / data_congestion['Truck Drivers'].sum() * 100
st.dataframe(data_congestion)
st.altair_chart(pie_chart(data_congestion, 'Truck Drivers', 'Experience', "Truck Driver Congestion Experience (Pie Chart)"))

# 6. Waiting Time per Visit
st.header("6. ⏱ Waiting Time at Gates")
data_wait = pd.DataFrame({
    'Time Category': ['<30 mins', '30 mins-1 hr', '1-2 hrs', '2-5 hrs', '>5 hrs'],
    'Truck Drivers': [87, 98, 119, 185, 245]
})
data_wait["Percentage"] = data_wait['Truck Drivers'] / data_wait['Truck Drivers'].sum() * 100
st.dataframe(data_wait)
st.altair_chart(pie_chart(data_wait, 'Truck Drivers', 'Time Category', "Waiting Time Distribution (Pie Chart)"))

# 7. Gate Usage Distribution
st.header("7. 🛣 Gate Usage Distribution")
data_gate = pd.DataFrame({
    'Gate': ['Gate 24', 'Gate 18', 'Main Gate 9/10', 'ICD', 'Others (12,13,15,16)'],
    'Truck Drivers': [475, 308, 93, 0, 0]
})
data_gate["Percentage"] = data_gate['Truck Drivers'] / data_gate['Truck Drivers'].sum() * 100
st.dataframe(data_gate)

fig7 = alt.Chart(data_gate).mark_bar().encode(
    x='Gate',
    y='Truck Drivers',
    color='Gate',
    tooltip=['Gate', 'Truck Drivers', 'Percentage']
).properties(title="Gate Usage Distribution")

labels7 = alt.Chart(data_gate).mark_text(
    dy=-10,
    size=13,
    fontWeight='bold',
    color='black',
    stroke='white',
    strokeWidth=2
).encode(
    x='Gate',
    y='Truck Drivers',
    text=alt.Text('Percentage:Q', format='.1f')
)

st.altair_chart(fig7 + labels7, use_container_width=True)

# 8. Congestion Time Frequency
st.header("8. 🕐 Time of Day with Most Congestion")
data_time_congestion = pd.DataFrame({
    'Time': ['Morning', 'Midday', 'Afternoon', 'Evening'],
    'Truck Drivers': [150, 219, 480, 368]
})
data_time_congestion["Percentage"] = data_time_congestion['Truck Drivers'] / data_time_congestion['Truck Drivers'].sum() * 100
st.dataframe(data_time_congestion)

fig8 = alt.Chart(data_time_congestion).mark_bar().encode(
    x='Time',
    y='Truck Drivers',
    color='Time',
    tooltip=['Time', 'Truck Drivers', 'Percentage']
).properties(title="Congestion Time by Day Period")

labels8 = alt.Chart(data_time_congestion).mark_text(
    dy=-10,
    size=13,
    fontWeight='bold',
    color='black',
    stroke='white',
    strokeWidth=2
).encode(
    x='Time',
    y='Truck Drivers',
    text=alt.Text('Percentage:Q', format='.1f')
)

st.altair_chart(fig8 + labels8, use_container_width=True)

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
st.dataframe(data_causes)

fig9 = alt.Chart(data_causes).mark_bar().encode(
    x='Cause',
    y='Truck Drivers',
    color='Cause',
    tooltip=['Cause', 'Truck Drivers', 'Percentage']
).properties(title="Causes of Traffic Congestion")

labels9 = alt.Chart(data_causes).mark_text(
    dy=-10,
    size=13,
    fontWeight='bold',
    color='black',
    stroke='white',
    strokeWidth=2
).encode(
    x='Cause',
    y='Truck Drivers',
    text=alt.Text('Percentage:Q', format='.1f')
)

st.altair_chart(fig9 + labels9, use_container_width=True)

# 10. Effects of Congestion
st.header("10. 📉 Effects of Congestion on Work")
data_effects = pd.DataFrame({
    'Effect': [
        'Longer working hours', 'Increased storage fees', 'Missed delivery schedules',
        'Increased fuel costs', 'Increased turnaround times'
    ],
    'Truck Drivers': [562, 386, 258, 305, 183]
})
data_effects["Percentage"] = data_effects['Truck Drivers'] / data_effects['Truck Drivers'].sum() * 100
st.dataframe(data_effects)

fig10 = alt.Chart(data_effects).mark_bar().encode(
    x='Effect',
    y='Truck Drivers',
    color='Effect',
    tooltip=['Effect', 'Truck Drivers', 'Percentage']
).properties(title="Effects of Congestion on Work")

labels10 = alt.Chart(data_effects).mark_text(
    dy=-10,
    size=13,
    fontWeight='bold',
    color='black',
    stroke='white',
    strokeWidth=2
).encode(
    x='Effect',
    y='Truck Drivers',
    text=alt.Text('Percentage:Q', format='.1f')
)

st.altair_chart(fig10 + labels10, use_container_width=True)
