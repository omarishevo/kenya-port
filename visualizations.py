import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set up the page
st.set_page_config(layout="wide", page_title="KPA Gate Comprehensive Dashboard")

# Title and sidebar
st.title("KPA Gate Operations Comprehensive Dashboard")
st.sidebar.title("Navigation")

# Create navigation
page = st.sidebar.radio("Select Analysis Page:", [
    "Stakeholder Overview",
    "Gender Distribution",
    "Work Experience",
    "Gate Visit Frequency",
    "Traffic Congestion",
    "Waiting Time Analysis",
    "Gate Utilization",
    "Congestion Timing",
    "Congestion Causes",
    "Work Impact Analysis"
])

# Page 1: Stakeholder Overview
if page == "Stakeholder Overview":
    st.header("FY2025/26 Stakeholder Overview - KPA Gate Analysis")
    
    # Data
    data = {
        'Category': ['Truck Driver', 'Clearing Agents', 'Custom Officials', 'KPA Staffs', 'Traffic Police'],
        'Total (n)': [714, 113, 10, 22, 9],
        '% Kenyans': [88.4, 96.5, 100, 100, 100]
    }

    regional_data_truck = {
        'Uganda': 6.2,
        'Tanzania': 2.2,
        'South Sudan': 0.3,
        'Rwanda': 1.0,
        'Burundi': 0.8,
        'DRC Congo': 1.1
    }

    regional_data_clearing = {
        'Uganda': 0.9,
        'Tanzania': 1.8,
        'Rwanda': 0.9
    }

    df = pd.DataFrame(data)

    # Display data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stakeholder Summary")
        st.dataframe(df)
        
        # Bar chart
        st.subheader("Total Count and % Kenyans per Category")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(df['Category'], df['Total (n)'], color='skyblue', label='Total (n)')
        ax2 = ax1.twinx()
        ax2.plot(df['Category'], df['% Kenyans'], color='orange', marker='o', label='% Kenyans')
        ax1.set_ylabel('Total Count')
        ax2.set_ylabel('% Kenyans')
        ax1.set_xticklabels(df['Category'], rotation=45)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        st.pyplot(fig)

    with col2:
        # Pie charts
        st.subheader("Regional Representation (Non-Kenyans)")
        
        st.markdown("Truck Drivers - Regional Breakdown")
        fig1, ax1 = plt.subplots()
        ax1.pie(regional_data_truck.values(), labels=regional_data_truck.keys(), autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)
        
        st.markdown("Clearing Agents - Regional Breakdown")
        fig2, ax2 = plt.subplots()
        ax2.pie(regional_data_clearing.values(), labels=regional_data_clearing.keys(), autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')
        st.pyplot(fig2)

# Page 2: Gender Distribution
elif page == "Gender Distribution":
    st.header("Gender Distribution Across Categories")
    
    # Data
    data = {
        "Category": ["Truck Drivers", "Clearing Agents", "Customs Officials", "KPA Staff"],
        "Male": [700, 119, 7, 14],
        "Female": [14, 5, 3, 8],
        "Total": [714, 124, 10, 22],
        "% Male": [98.0, 96.0, 70.0, 63.6],
        "% Female": [2.0, 4.0, 30.0, 36.4]
    }

    df = pd.DataFrame(data)

    # Display the raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart", "Pie Charts", "Percentage Comparison", "Stacked Bar Chart"])

    with tab1:
        st.subheader("Count by Gender and Category")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_melted = df.melt(id_vars="Category", value_vars=["Male", "Female"], 
                            var_name="Gender", value_name="Count")
        sns.barplot(data=df_melted, x="Category", y="Count", hue="Gender", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Count of Males and Females by Category")
        st.pyplot(fig)

    with tab2:
        st.subheader("Gender Distribution per Category")
        cols = st.columns(2)
        
        for i, category in enumerate(df["Category"]):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 6))
                sizes = [df.loc[i, "% Male"], df.loc[i, "% Female"]]
                labels = ["Male", "Female"]
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title(f"{category} Gender Distribution")
                st.pyplot(fig)

    with tab3:
        st.subheader("Percentage Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_melted_pct = df.melt(id_vars="Category", value_vars=["% Male", "% Female"], 
                                var_name="Gender", value_name="Percentage")
        sns.barplot(data=df_melted_pct, x="Category", y="Percentage", hue="Gender", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Percentage of Males and Females by Category")
        ax.set_ylim(0, 100)
        st.pyplot(fig)

    with tab4:
        st.subheader("Stacked Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        df[["Category", "Male", "Female"]].set_index("Category").plot(
            kind="bar", stacked=True, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Stacked Count of Males and Females by Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Page 3: Work Experience
elif page == "Work Experience":
    st.header("Work Experience Distribution Across Categories")
    
    # Data
    data = {
        "Category": ["Truck Driver", "Clearing Agents", "Custom Officials", "KPA Staff Traffic Police", "Unknown"],
        "less than 1 year": [43, 3, 11, 2, 4],
        "1-5 years": [210, 2, 50, 1, 5],
        "6-10 years": [133, 1, 25, 8, 0],
        "over 10 years": [328, 4, 38, 11, 0],
        "% less than 1 year": [6.0, 30.0, 8.9, 9.1, 44.4],
        "% 1-5 years": [29.4, 20.0, 40.3, 4.5, 55.6],
        "% 6-10 years": [18.6, 10.0, 20.2, 36.4, 0.0],
        "% over 10 years": [45.9, 40.0, 30.6, 50.0, 0.0]
    }

    df = pd.DataFrame(data)

    # Display the raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Count Bar Chart", "Percentage Heatmap", "Stacked Percentage", "Experience Distribution"])

    with tab1:
        st.subheader("Count by Experience Level and Category")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        count_cols = ["less than 1 year", "1-5 years", "6-10 years", "over 10 years"]
        df_melted = df.melt(id_vars="Category", value_vars=count_cols, 
                            var_name="Experience", value_name="Count")
        
        sns.barplot(data=df_melted, x="Category", y="Count", hue="Experience", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Count of Employees by Experience Level and Category")
        ax.legend(title="Experience Level")
        st.pyplot(fig)

    with tab2:
        st.subheader("Percentage Distribution Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pct_cols = ["% less than 1 year", "% 1-5 years", "% 6-10 years", "% over 10 years"]
        pct_data = df[["Category"] + pct_cols].set_index("Category")
        pct_data.columns = [col.replace("% ", "") for col in pct_data.columns]
        
        sns.heatmap(pct_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        ax.set_title("Percentage Distribution of Experience Levels by Category")
        ax.set_xlabel("Experience Level")
        st.pyplot(fig)

    with tab3:
        st.subheader("Stacked Percentage Bar Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pct_data.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xticklabels(pct_data.index, rotation=45)
        ax.set_ylabel("Percentage")
        ax.set_title("Percentage Distribution of Experience Levels (Stacked)")
        ax.legend(title="Experience Level", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    with tab4:
        st.subheader("Experience Distribution per Category")
        
        cols = st.columns(2)
        experience_levels = ["less than 1 year", "1-5 years", "6-10 years", "over 10 years"]
        colors = sns.color_palette("pastel")[:4]
        
        for i, category in enumerate(df["Category"]):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 6))
                sizes = [df.loc[i, f"% {level}"] for level in experience_levels]
                
                if sum(sizes) > 0:
                    ax.pie(sizes, labels=experience_levels, autopct='%1.1f%%', 
                          colors=colors, startangle=90)
                    ax.set_title(f"{category} Experience Distribution")
                    st.pyplot(fig)
                else:
                    st.write(f"No data for {category}")

# Page 4: Gate Visit Frequency
elif page == "Gate Visit Frequency":
    st.header("Gate Visit Frequency Analysis")
    
    # Data
    data = {
        "Category": ["TRUCK DRIVER", "CLEARING AGENTS", "CUSTOM OFFICIAL", "TRAFFIC POLICE", "KPA STAFF"],
        "DAILY": [285, 80, 4, 9, 8],
        "SEVERAL TIMES A WEEK(2-3TIMES)": [176, 35, 3, 0, 14],
        "ONCE A WEEK": [105, 5, 1, 0, 0],
        "A FEW TIMES A MONTH (1-3 TIMES)": [138, 4, 1, 0, 0],
        "RARELY LESS THAN ONCE A MONTH": [10, 0, 1, 0, 0]
    }

    percent_data = {
        "Category": ["TRUCK DRIVER", "CLEARING AGENTS", "CUSTOM OFFICIAL", "TRAFFIC POLICE", "KPA STAFF"],
        "DAILY": [39.9, 64.5, 40, 100, 36.4],
        "SEVERAL TIMES A WEEK(2-3TIMES)": [24.6, 28.2, 30, 0, 63.6],
        "ONCE A WEEK": [14.7, 4, 10, 0, 0],
        "A FEW TIMES A MONTH (1-3 TIMES)": [19.3, 3.2, 10, 0, 0],
        "RARELY LESS THAN ONCE A MONTH": [1.4, 0, 10, 0, 0]
    }

    df_counts = pd.DataFrame(data)
    df_percents = pd.DataFrame(percent_data)

    # Melt the data
    melted_counts = df_counts.melt(id_vars="Category", var_name="Frequency", value_name="Count")
    melted_percents = df_percents.melt(id_vars="Category", var_name="Frequency", value_name="Percentage")
    combined_df = melted_counts.merge(melted_percents, on=["Category", "Frequency"])

    # Visualization section
    tab1, tab2, tab3, tab4 = st.tabs(["Stacked Bar Chart (Counts)", "Stacked Bar Chart (Percentages)", 
                                      "Heatmap", "Grouped Bar Charts"])

    with tab1:
        st.subheader("Visit Frequency by Category (Counts)")
        fig, ax = plt.subplots(figsize=(12, 6))
        melted_counts.pivot(index="Category", columns="Frequency", values="Count").plot(
            kind='bar', stacked=True, ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Number of Visits")
        plt.xlabel("Category")
        plt.legend(title="Visit Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Visit Frequency by Category (Percentages)")
        fig, ax = plt.subplots(figsize=(12, 6))
        melted_percents.pivot(index="Category", columns="Frequency", values="Percentage").plot(
            kind='bar', stacked=True, ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Percentage (%)")
        plt.xlabel("Category")
        plt.legend(title="Visit Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Visit Frequency Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_counts = melted_counts.pivot(index="Category", columns="Frequency", values="Count")
        sns.heatmap(pivot_counts, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        plt.xticks(rotation=45)
        plt.title("Visit Counts by Category and Frequency")
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("Grouped Bar Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Counts")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=melted_counts, x="Category", y="Count", hue="Frequency", ax=ax)
            plt.xticks(rotation=45)
            plt.ylabel("Number of Visits")
            plt.xlabel("Category")
            plt.legend(title="Visit Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("Percentages")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=melted_percents, x="Category", y="Percentage", hue="Frequency", ax=ax)
            plt.xticks(rotation=45)
            plt.ylabel("Percentage (%)")
            plt.xlabel("Category")
            plt.legend(title="Visit Frequency", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)

# Page 5: Traffic Congestion
elif page == "Traffic Congestion":
    st.header("Traffic Congestion Experience Analysis")
    
    # Data
    data = {
        "Category": ["TRUCK DRIVER", "CLEARING AGENTS"],
        "NEVER": [15, 0],
        "RARELY": [48, 6],
        "SOMETIMES": [216, 58],
        "OFTEN": [190, 39],
        "ALWAYS": [245, 21]
    }

    percent_data = {
        "Category": ["TRUCK DRIVER", "CLEARING AGENTS"],
        "NEVER": [2.1, 0],
        "RARELY": [6.7, 4.8],
        "SOMETIMES": [30.3, 46.8],
        "OFTEN": [26.6, 31.5],
        "ALWAYS": [34.3, 16.9]
    }

    df_counts = pd.DataFrame(data)
    df_percents = pd.DataFrame(percent_data)

    # Melt the data
    melted_counts = df_counts.melt(id_vars="Category", var_name="Experience", value_name="Count")
    melted_percents = df_percents.melt(id_vars="Category", var_name="Experience", value_name="Percentage")
    combined_df = melted_counts.merge(melted_percents, on=["Category", "Experience"])

    # Order the experience levels
    experience_order = ["NEVER", "RARELY", "SOMETIMES", "OFTEN", "ALWAYS"]
    melted_counts["Experience"] = pd.Categorical(melted_counts["Experience"], categories=experience_order, ordered=True)
    melted_percents["Experience"] = pd.Categorical(melted_percents["Experience"], categories=experience_order, ordered=True)

    # Visualization section
    tab1, tab2, tab3, tab4 = st.tabs(["Stacked Bar Chart (Counts)", "Stacked Bar Chart (Percentages)", 
                                      "Heatmap", "Grouped Bar Charts"])

    with tab1:
        st.subheader("Traffic Experience by Category (Counts)")
        fig, ax = plt.subplots(figsize=(10, 6))
        melted_counts.pivot(index="Category", columns="Experience", values="Count").plot(
            kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.xticks(rotation=0)
        plt.ylabel("Number of Respondents")
        plt.xlabel("Category")
        plt.legend(title="Traffic Experience", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Traffic Experience by Category (Percentages)")
        fig, ax = plt.subplots(figsize=(10, 6))
        melted_percents.pivot(index="Category", columns="Experience", values="Percentage").plot(
            kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.xticks(rotation=0)
        plt.ylabel("Percentage (%)")
        plt.xlabel("Category")
        plt.legend(title="Traffic Experience", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Traffic Experience Heatmap")
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot_counts = melted_counts.pivot(index="Category", columns="Experience", values="Count")
        sns.heatmap(pivot_counts, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        plt.title("Traffic Experience Counts by Category")
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("Grouped Bar Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Counts")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=melted_counts, x="Category", y="Count", hue="Experience", 
                        palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], ax=ax)
            plt.ylabel("Number of Respondents")
            plt.xlabel("Category")
            plt.legend(title="Traffic Experience", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("Percentages")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=melted_percents, x="Category", y="Percentage", hue="Experience", 
                        palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], ax=ax)
            plt.ylabel("Percentage (%)")
            plt.xlabel("Category")
            plt.legend(title="Traffic Experience", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)

# Page 6: Waiting Time Analysis
elif page == "Waiting Time Analysis":
    st.header("Waiting Time at KPA Gates per Visit")
    
    # Data
    data = {
        "Category": ["TRUCK DRIVER", "CLEARING AGENTS"],
        "Less than 30 mins": [87, 27],
        "30 mins-1 hr": [98, 23],
        "1-2 hrs": [119, 26],
        "2-5 hrs": [185, 24],
        "Over 5 hours": [225, 24]
    }

    percent_data = {
        "Category": ["TRUCK DRIVER", "CLEARING AGENTS"],
        "Less than 30 mins": [12.2, 21.8],
        "30 mins-1 hr": [13.7, 18.5],
        "1-2 hrs": [16.7, 21.0],
        "2-5 hrs": [25.9, 19.4],
        "Over 5 hours": [31.5, 19.4]
    }

    df_counts = pd.DataFrame(data)
    df_percents = pd.DataFrame(percent_data)

    # Melt the data
    melted_counts = df_counts.melt(id_vars="Category", var_name="Waiting Time", value_name="Count")
    melted_percents = df_percents.melt(id_vars="Category", var_name="Waiting Time", value_name="Percentage")

    # Order the waiting time categories
    waiting_order = ["Less than 30 mins", "30 mins-1 hr", "1-2 hrs", "2-5 hrs", "Over 5 hours"]
    melted_counts["Waiting Time"] = pd.Categorical(melted_counts["Waiting Time"], categories=waiting_order, ordered=True)
    melted_percents["Waiting Time"] = pd.Categorical(melted_percents["Waiting Time"], categories=waiting_order, ordered=True)

    # Visualization section
    tab1, tab2, tab3, tab4 = st.tabs(["Stacked Bar Charts", "Percentage Distribution", 
                                      "Comparative Analysis", "Raw Data"])

    with tab1:
        st.subheader("Waiting Time Distribution (Counts)")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        df_counts.set_index("Category").T.plot(kind='bar', stacked=True, ax=ax1, 
                                             color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'])
        ax1.set_title("Stacked Bar Chart (Counts)")
        ax1.set_ylabel("Number of Respondents")
        ax1.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        df_percents.set_index("Category").T.plot(kind='bar', stacked=True, ax=ax2,
                                              color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'])
        ax2.set_title("Stacked Bar Chart (Percentages)")
        ax2.set_ylabel("Percentage (%)")
        ax2.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Percentage Distribution by Category")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Truck Drivers")
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(
                df_percents.loc[0, "Less than 30 mins":"Over 5 hours"],
                labels=waiting_order,
                autopct='%1.1f%%',
                colors=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'],
                startangle=90,
                wedgeprops=dict(width=
