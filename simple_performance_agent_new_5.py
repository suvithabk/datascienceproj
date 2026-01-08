
import os
import json
from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage , AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
# from openai import BaseModel
from pydantic import BaseModel
import uuid

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime


# Load env vars
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# Read your dataset
df = pd.read_csv("./data/performance_data.csv")
df.columns = df.columns.str.strip().str.lower()
# Clean mmyy and percent unhealthy
df['mmyy'] = df['mmyy'].str.strip().str.lower()
df['percent unhealthy'] = df['percent unhealthy'].str.replace('%', '', regex=False).astype(float)
df['scenario']=df['scenario'].str.strip().str.lower()

# Ensure issue columns are numeric (coerce any stray strings)
for col in ['server issues', 'client issues', 'latency issues', 'total clicks', 'unhealthy clicks']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

class ChartItem(BaseModel):
    id: str = str(uuid.uuid4())   # unique ID for each chart
    fig_json: str 

# State
class PlotData(BaseModel):
    charts: List[ChartItem]


# State
class AgentState(TypedDict):
    question: str
    summary_data: str
    raw_data: str
    buyer_data:str
    buyer_list:List[str]
    scenario_data:str
    llm_response: str
    charts: PlotData
    messages: List[BaseMessage]

# === Data Processing Node ===
def summarize_overall():
    print("In function summarize_overall")
    summary = df.groupby("mmyy").agg({
        "total clicks": "sum",
        "unhealthy clicks": "sum",      
        "server issues": "sum",
        "client issues": "sum",
        "latency issues": "sum"
    }).reset_index()
    summary["percent unhealthy"] = (
        summary["unhealthy clicks"] * 100/ summary["total clicks"] 
    ).round(2)

    summary["percent server issues"] = (summary["server issues"] / summary["total clicks"] * 100).round(2)
    summary["percent client issues"] = (summary["client issues"] / summary["total clicks"] * 100).round(2)
    summary["percent latency issues"] = (summary["latency issues"] / summary["total clicks"] * 100).round(2)
    print("Output of summarize_overall :" , summary)
    return summary

def compare_months(m1, m2):
    print("In function compare_months")

    full = summarize_overall()
    m1_df = full[full["mmyy"] == m1]
    m2_df = full[full["mmyy"] == m2]
    if m1_df.empty or m2_df.empty:
        return None
    return pd.concat([m1_df, m2_df])

def compare_buyers_between_months(m1, m2):
    print("In function compare_buyers_between_months")

    # Step 1: Filter data to the selected months
    filtered = df[df["mmyy"].isin([m1.lower(), m2.lower()])]

    # Step 2: Group by buyer and month, aggregating both metrics
    grouped = (
        filtered.groupby(["buyer", "mmyy"], as_index=False)
        .agg({
            "total clicks": "sum",
            "unhealthy clicks": "sum"
        })
    )

    grouped["percent_unhealthy"] = (
        grouped["unhealthy clicks"] *100/ grouped["total clicks"]
    ).fillna(0)

        # Step 2: Get buyer counts by month
    buyer_month_counts = grouped.groupby("buyer")["mmyy"].nunique()

    # Step 3: Keep only buyers with BOTH months
    valid_buyers = buyer_month_counts[buyer_month_counts == 2].index
    grouped = grouped[grouped["buyer"].isin(valid_buyers)]

    # Step 3: Pivot total clicks
    total_clicks_pivot = grouped.pivot(index="buyer", columns="mmyy", values="total clicks").fillna(0)
    total_clicks_pivot.columns = [f"clicks_{col}" for col in total_clicks_pivot.columns]

    # Step 4: Pivot unhealthy clicks
    unhealthy_clicks_pivot = grouped.pivot(index="buyer", columns="mmyy", values="unhealthy clicks").fillna(0)
    unhealthy_clicks_pivot.columns = [f"unhealthy_{col}" for col in unhealthy_clicks_pivot.columns]


    percent_unhealthy_pivot = grouped.pivot(index="buyer", columns="mmyy", values="percent_unhealthy").fillna(0)
    percent_unhealthy_pivot.columns = [f"percent_unhealthy_{col}" for col in percent_unhealthy_pivot.columns]
    # Step 5: Combine the two
    result = total_clicks_pivot.join(unhealthy_clicks_pivot).join(percent_unhealthy_pivot)

    # Step 6: Calculate deltas
    result["click_delta"] = result[f"clicks_{m2.lower()}"] - result[f"clicks_{m1.lower()}"]
    result["unhealthy_delta"] = result[f"unhealthy_{m2.lower()}"] - result[f"unhealthy_{m1.lower()}"]
    result["percent_unhealthy_delta"] = (
    result[f"percent_unhealthy_{m2.lower()}"] - result[f"percent_unhealthy_{m1.lower()}"]
    )
    # Step 7: Identify top growing buyer by total clicks
    top_buyer = result["click_delta"].idxmax()
    top_growth = result.loc[top_buyer, "click_delta"]
    print("top buyer : ", top_buyer)
    print("top growth : ", top_growth)

    return result.reset_index(), top_buyer, top_growth

def applynormalization(group):
    # Normalize unhealthy clicks
    print(group)
    min_uc = group['unhealthy_clicks'].min()
    max_uc = group['unhealthy_clicks'].max()
    if max_uc - min_uc == 0:
        group['unhealthy_clicks_normalized'] = 0
    else:
        group['unhealthy_clicks_normalized'] = (group['unhealthy_clicks'] - min_uc) / (max_uc - min_uc)
    print('gtoup')
    print(group)

    if group['percent_unhealthy'].dtype == object:
        group['percent_unhealthy'] = group['percent_unhealthy'].str.rstrip('%').astype(float)
    
    min_rate = group['percent_unhealthy'].min()
    max_rate = group['percent_unhealthy'].max()
    if max_rate - min_rate == 0:
        group['unhealthy_clicks_percent_normalized'] = 0
    else:
        group['unhealthy_clicks_percent_normalized'] = (group['percent_unhealthy'] - min_rate) / (max_rate - min_rate)
    
    group['weighted_normalized_score'] = (0.45 * group['unhealthy_clicks_normalized']) + (0.55 * group['unhealthy_clicks_percent_normalized'])
    print(group)
    return group


def compare_scenario_between_months(m1, m2):
    print("In function compare_scenario_between_months")

    # Step 1: Filter data to the selected months
    filtered = df[df["mmyy"].isin([m1.lower(), m2.lower()])]

    # Step 2: Group by scenario and month, aggregating both metrics
    grouped = (
        filtered.groupby(["scenario", "mmyy"], as_index=False)
        .agg({
            "total clicks": "sum",
            "unhealthy clicks": "sum"
        })
    )

    grouped["percent_unhealthy"] = (
        grouped["unhealthy clicks"] *100/ grouped["total clicks"]
    ).fillna(0)

        # Step 2: Get scenario counts by month
    scenario_month_counts = grouped.groupby("scenario")["mmyy"].nunique()

    # Step 3: Keep only buyers with BOTH months
    valid_scenario = scenario_month_counts[scenario_month_counts == 2].index
    grouped = grouped[grouped["scenario"].isin(valid_scenario)]

    # Step 3: Pivot total clicks
    total_clicks_pivot = grouped.pivot(index="scenario", columns="mmyy", values="total clicks").fillna(0)
    total_clicks_pivot.columns = [f"clicks_{col}" for col in total_clicks_pivot.columns]

    # Step 4: Pivot unhealthy clicks
    unhealthy_clicks_pivot = grouped.pivot(index="scenario", columns="mmyy", values="unhealthy clicks").fillna(0)
    unhealthy_clicks_pivot.columns = [f"unhealthy_{col}" for col in unhealthy_clicks_pivot.columns]


    percent_unhealthy_pivot = grouped.pivot(index="scenario", columns="mmyy", values="percent_unhealthy").fillna(0)
    percent_unhealthy_pivot.columns = [f"percent_unhealthy_{col}" for col in percent_unhealthy_pivot.columns]

    # Step 5: Combine the two
    result = total_clicks_pivot.join(unhealthy_clicks_pivot).join(percent_unhealthy_pivot)

    # Step 6: Calculate deltas
    result["click_delta"] = result[f"clicks_{m2.lower()}"] - result[f"clicks_{m1.lower()}"]
    result["unhealthy_delta"] = result[f"unhealthy_{m2.lower()}"] - result[f"unhealthy_{m1.lower()}"]
    result["percent_unhealthy_delta"] = (
    result[f"percent_unhealthy_{m2.lower()}"] - result[f"percent_unhealthy_{m1.lower()}"]
    )
    # Step 7: Identify top growing buyer by total clicks
    top_scenario= result["click_delta"].idxmax()
    top_growth = result.loc[top_scenario, "click_delta"]
    print("top top_scenario : ", top_scenario)
    print("top growth : ", top_growth)
    return result.reset_index(), top_scenario, top_growth

def prepare_data(state: AgentState):
    question = state["question"].lower()
    months = [m for m in df["mmyy"].unique()]
    mentioned_months = [m for m in months if m in question]
    print(mentioned_months)

    buyers = [b.lower() for b in df["buyer"].dropna().unique()]
    mentioned_buyers = [b for b in buyers if b in question]
    scenario=[s.lower() for s in df["scenario"].dropna().unique()]
    mentioned_scenarios=[s for s in scenario if s in question]
    if not mentioned_buyers:
        mentioned_buyers = buyers
    if not mentioned_scenarios:
        mentioned_scenarios = scenario

    print("mentioned_buyers are :" , mentioned_buyers)
    print('mentioned_scenarios')
    print(mentioned_scenarios)


    summary_text = ""

    buyer_df_processed = pd.DataFrame()
    scenario_df = pd.DataFrame()
    filtered_df=pd.DataFrame()
    raw_text = df.to_markdown(index=False)
    compare_df=pd.DataFrame()
    filtered_df=pd.DataFrame()
    overall_df=pd.DataFrame()

    ## Added logic when asked about one month
    if len(mentioned_months) == 1 and "scenario" in question:
        target_m = mentioned_months[0]
        one_month = (
            df[df["mmyy"] == target_m]
            .groupby("scenario", as_index=False)
            .agg({"total clicks": "sum", "unhealthy clicks": "sum"})
        )
        one_month["percent_unhealthy"] = (
            (one_month["unhealthy clicks"] / one_month["total clicks"]) * 100
        ).fillna(0).round(2)
        one_month["month"] = target_m

        # Leader
        top_row = one_month.sort_values("unhealthy clicks", ascending=False).head(1)
        top_scenario = top_row["scenario"].iat[0]
        top_val = int(top_row["unhealthy clicks"].iat[0])

        summary_text = one_month.to_markdown(index=False)
        summary_text += f"\n\nIn {target_m}, the scenario with the highest total unhealthy clicks is **{top_scenario}** ({top_val})."

        filtered_df = one_month   
    # Case: Buyer-level MoM comparison
    elif len(mentioned_months) == 2 and ("buyer" in question or "buyers" in question):
        print("Executing compare_buyers_between_months function")
        buyer_df, top_buyer, top_growth = compare_buyers_between_months(mentioned_months[0], mentioned_months[1])
        summary_text = buyer_df.to_markdown(index=False)
        print(buyer_df)
        summary_text += f"\n\nTop growing buyer: {top_buyer} (+{top_growth} clicks)"
        print("in here------")
        print("op of function compare_buyers_between_months", buyer_df)
        buyer_df = buyer_df[buyer_df["buyer"].isin(mentioned_buyers)]
        print('filtered df')
        print(buyer_df)
        buyer_df.columns = buyer_df.columns.str.replace(" ", "_")
        # buyer_df = buyer_df.reset_index()
        buyer_df_processed = buyer_df.melt(
            id_vars=['buyer'],
            var_name='metric_month',
            value_name='value'
        )
        print('after melt')
        print(buyer_df_processed)
        split_cols = buyer_df_processed["metric_month"].str.rsplit("_", n=1, expand=True)
        buyer_df_processed["metric"] = split_cols[0]      # clicks / unhealthy / percent_unhealthy
        buyer_df_processed["month"] = split_cols[1]  
        
        buyer_df_processed = buyer_df_processed[
             buyer_df_processed["month"].notna()
             & ~buyer_df_processed["month"].str.contains("delta", na=False)
         ]
        df_long = pd.DataFrame(buyer_df_processed)

        buyer_df_processed = df_long.pivot_table(
                index=["month", "buyer"],
                columns="metric",
                values="value",
                aggfunc="first"
            ).reset_index()
        buyer_df_processed = buyer_df_processed.rename(columns={
            "clicks": "total_clicks",
            "unhealthy": "unhealthy_clicks"
        })
        print(buyer_df_processed)
        buyer_df_processed = buyer_df_processed.groupby("month").apply(applynormalization,include_groups=False).reset_index(drop=True)
        # buyer_df_processed=calculateweighted_score(buyer_df_processed)
        # buyer_df_processed['weighted_normalized_score'] = (0.45 * buyer_df_processed['unhealthy_clicks_percent_normalized']) + (0.55 * buyer_df_processed['unhealthy_clicks_normalized'])
        print('after calculation')
        print(buyer_df_processed)

    
    elif len(mentioned_months)==2 and "scenario" in question:
        print("Executing compare_scenario_between_months function")
        scenario_df, top_scenario, top_growth = compare_scenario_between_months(mentioned_months[0], mentioned_months[1])
        scenario_df_fromllm = llm_decide_output(
            scenario_df,"scenario",
            state["question"],
            mentioned_months,
            model,
            top_n=5
        )
        print('scenario_df_processed')
        print(scenario_df_fromllm)
        summary_text = scenario_df_fromllm.to_markdown(index=False)
        summary_text += f"\n\nTop growing scenario: {top_scenario} (+{top_growth} clicks)"
        filtered_df = scenario_df_fromllm[scenario_df_fromllm["scenario"].isin(mentioned_scenarios)]
        filtered_df.columns = filtered_df.columns.str.replace(" ", "_")
        scenario_df_processed = filtered_df.melt(
             id_vars=["scenario", "month"], value_name="value"
         )
        
        if "metric_month" in scenario_df_processed.columns:
            split_cols = scenario_df_processed["metric_month"].str.split("_", n=1, expand=True)
            if split_cols.shape[1] == 1:
                split_cols[1] = None
            scenario_df_processed["metric"] = split_cols[0]
            scenario_df_processed["month"] = split_cols[1]

    # Case: Overall month comparison
    elif len(mentioned_months) == 2:
        print("Executing compare_months function. This then goes to summarize_overall function to group by respective months")

        comp_df = compare_months(mentioned_months[0], mentioned_months[1])
        if comp_df is not None:
            summary_text = comp_df.to_markdown(index=False)
    


    # Case: fallback summary
    else:
        print("Executing  summarize_overall function ")

        overall = summarize_overall()
        summary_text = overall.to_markdown(index=False)

    # Always include raw data
    raw_text = df.to_markdown(index=False)
  
    return {
        "summary_data": summary_text,
        "raw_data": raw_text,
        "buyer_data":buyer_df_processed.to_json(orient="records") if not buyer_df_processed.empty else None,
        "scenario_data": filtered_df.to_json(orient="records") if not filtered_df.empty and len(mentioned_months) == 2 else None,
        "compare_data":compare_df.to_json(orient="records") if not compare_df.empty else None,
        "overall_data":overall_df.to_json(orient="records") if not compare_df.empty else None,
        "question": state["question"]
    }



def llm_decide_output(scenario_df,groupby_col, question, mentioned_months, llm):
  
    """
    Ask the LLM whether to return 'worst scenarios' or 'normal comparison' or 'best scenarios'
    and return a DataFrame ready for plotting.
    """
    # Convert scenario_df to dict for LLM
    scenario_summary = scenario_df.to_dict(orient="records")
   
    
    prompt = f"""
    You are a data analyst. The user asked: "{question}".
    
    Here is the scenario-level summary data:
    {scenario_summary}
    
    Instructions:
    - Convert dataframe to json format using "{groupby_col}" as the key for each row:

  
    {{
        "{mentioned_months[0]}": [{{"{groupby_col}": "...", "unhealthy_clicks":..,"total_clicks":..}}, ...],
        "{mentioned_months[1]}": [{{"{groupby_col}": "...","unhealthy_clicks":..,"total_clicks":..}}, ...]
    }}
    - Return only the JSON format and no extra text
    - Return json with no name for the json returned
    - Return the json with all rows of the dataframe
    """
    print('scenario df')
    print(scenario_df)
    response = llm.invoke([HumanMessage(content=prompt)]).content
    print(response)
    llm_json = json.loads(response)
    print('llmjson')
    print(llm_json)

# Convert to DataFrame
    df = pd.DataFrame(
        [
            {"month": month, groupby_col: item[groupby_col],"unhealthy_clicks":item["unhealthy_clicks"],"total_clicks":item["total_clicks"]}
            for month, items in llm_json.items()
            for item in items
        ]
    )
    return df


# === LLM Node ===
def ask_llm(state: AgentState):
    history=state.get("messages" , [])
    summary_text = state["summary_data"]
    print("summary_text" , summary_text)
    question = state["question"]
    raw_data=state["raw_data"]
    buyer_data=state["buyer_data"]
    scenario_data=state["scenario_data"]
 

    prompt = f"""
You are a precise and helpful data analyst. 

Below is performance data in two formats:

### Summarized Data (grouped by buyer and month):
{summary_text}

### Raw Data (each row as in the original file):
{raw_data}



Answer the user's question using ONLY the summarized table above.
Do not try to recalculate from the raw data if the summary already contains the answer.
Always trust the summarized table for totals, unhealthy clicks, and percent unhealthy.
For example :
If I ask you how many unhealthy clicks a buyer had in a month in mmyy you will do a sum across all rows where mmyy is that month and Buyer is the buyer mentioned
For example if i ask you How many total unhealthy clicks did iron man have in june? You will sum unhealthy clicks where mmyy is june and buyer is iron man
Be accurate.

If I ask you how many unhealthy clicks a scenario had in a month in mmyy you will do a sum across all rows where mmyy is that month and scenario is the scenario mentioned
For example if i ask you How many total unhealthy clicks did login have in june? You will sum unhealthy clicks where mmyy is june and scenario is login


When asked total unhealthy clicks by scenario or buyer always add together total unhealthy clicks grouped by that buyer or scenario
When asked total clicks by scenario or buyer always add together total clicks grouped by that buyer or scenario


When calculating percent unhealthy always use (sum(unhealthy clicks)/sum(total clicks) )*100

When calculating total unhealthy clicks across rows sum the values correctly
When calculating total clicks across rows sum the values correctly

You are good in Mathematics so Add values together  correctly
Only give your analysis or final answer. Do not give calculations unless asked

when asked for  highest total unhealthy clicks , find the scenario that has the highest total unhealthy clicks
Worst scenario or buyer is the one that has the highest number of total unhealthy clicks
when asked about  server issues or client issues or  and latency issues always sum the values across rows
User's question: {question}
"""

    # messages = [
    #     SystemMessage(content="You are a strict, accurate data analysis assistant."),
    #     HumanMessage(content=prompt)
    # ]
    system_msg=SystemMessage(content="You are a strict, accurate data analysis assistant.")
    user_msg= HumanMessage(content=prompt)
    response = model.invoke(history + [system_msg, user_msg])
    new_history = history + [system_msg, user_msg, AIMessage(content=response.content)]
    print("llm response is" ,response )
    return {
        "llm_response": response.content,
        "messages": new_history
    }

def generate_charts_node(state: AgentState):
    """
    Generates Plotly charts based on the user's task and the provided CSV data.
    """
    task = state["question"].lower()
    chart_figures = []

    if(state["scenario_data"]):
        scneario_df=json.loads(state["scenario_data"])
        print('from charts function')
        print(scneario_df)
        df_scenario = pd.DataFrame(scneario_df)
        df_scenario.columns = df_scenario.columns.str.strip().str.lower().str.replace(" ", "_")

        print('df')
        print(df_scenario)
        month_order = sorted(df_scenario['month'].unique())
        
        df_scenario_long = df_scenario.melt(
                        id_vars=["month", "scenario"],
                        value_vars=[ "unhealthy_clicks"],
                        var_name="value_type",
                        value_name="value"
                    )
        fig = px.bar(
                            df_scenario_long,
                            x="month",
                            y="value",
                            color="scenario",
                            barmode="group",
                            category_orders={"month": ["may", "june"]},
                            title=f"Scenario Comparison By Unhealthy Clicks"
                        )

        fig.update_yaxes(matches='y')
        fig.update_yaxes(title_text='Unhealthy Clicks', col=1)              
        fig.update_layout(xaxis_title='Month')
        chart_figures.append(
                    ChartItem(id=str(uuid.uuid4()), fig_json=fig.to_json())
                )
        # chart_figures.append(fig.to_json())

    current_month_str = datetime.now().strftime("%b")
   
    if "august unhealthy clicks by buyer" in task:
        if "mmyy" in df.columns:
            df_current = df[df["mmyy"] == current_month_str]
            print(df_current)
            if not df_current.empty:
                # Group by 'BUYER' and sum latency to find the worst 5
                top_5_buyers = df_current.groupby("buyer")["unhealthy clicks"].sum().nlargest(5).index
                df_filtered = df_current[df_current["buyer"].isin(top_5_buyers)]
                
                # Generate a bar chart for the top 5 buyers for the current month
                fig = px.bar(
                    df_filtered,
                    x="buyer",
                    y="unhealthy clicks",
                    color="buyer",
                    title=f"Top 5 Worst Buyers by Unhealthy Clicks for the Current Month"
                )
                # chart_figures.append(fig.to_json())
                chart_figures.append(
                    ChartItem(id=str(uuid.uuid4()), fig_json=fig.to_json())
                )
        else:
            print("Warning: 'mmyy' column not found for current month latency chart.")

    elif "unhealthy clicks by buyer for may and june" in task:
        df_grouped = df.groupby(["buyer", "mmyy"], as_index=False)["unhealthy clicks"].sum()
        df_filtered = (
            df_grouped.groupby("mmyy", group_keys=False)
            .apply(lambda g: g.nlargest(5, "unhealthy clicks"))
            .reset_index(drop=True)
        )
        print(df_filtered)
        print('month over month unhealthy clicks by buyer')
 
        fig = px.bar(
            df_filtered,
            x="mmyy",
            y="unhealthy clicks",
            color="buyer",
            barmode="group",   # grouped bars under each month
            title="Top 5 Buyers per Month",
        )
 
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Unhealthy Clicks",
            legend_title="Buyer",)
        chart_figures.append(
                    ChartItem(id=str(uuid.uuid4()), fig_json=fig.to_json())
                )
    
    return {"charts": PlotData(charts=chart_figures)}
# === Build Graph ===
builder = StateGraph(AgentState)
builder.add_node("prepare_data", prepare_data)
builder.add_node("ask_llm", ask_llm)
builder.add_node("generate_charts_node",generate_charts_node)

builder.set_entry_point("prepare_data")
builder.add_edge("prepare_data", "ask_llm")
builder.add_edge("ask_llm", "generate_charts_node")
builder.add_edge("generate_charts_node", END)


import numpy as np

import streamlit as st
with SqliteSaver.from_conn_string(":memory:") as memory:
    graph = builder.compile(checkpointer=memory)


    # === Run Locally ===
    if __name__ == "__main__":
        print("🤖 Performance Q&A Agent")
        df2 = pd.read_csv("./data/functional_scenario_data.csv")
        df2.columns = df2.columns.str.strip().str.lower()
        if "mmyy" in df2.columns:
            df2["mmyy"] = df2["mmyy"].astype(str).str.strip().str.lower()

        st.title("Performance Q&A Assistant")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []  # list of HumanMessage / AIMessage
        if "qa_log" not in st.session_state:
            st.session_state.qa_log = [] 
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = "user-1"  # set per user/session as you like

        if "functional_table" not in st.session_state:
            st.session_state.functional_table = None

        if st.session_state.functional_table is not None:
            st.subheader("Functional Scenario Data")
            st.dataframe(st.session_state.functional_table, use_container_width=True)

        if st.session_state.qa_log:
            for entry in st.session_state.qa_log:
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f"**A:** {entry['answer']}")
                                # Show charts if available
                for chart in entry.get("charts", []):
                    fig_json = chart.get("fig_json")  
                    chart_id = chart.get("id")
                    if fig_json:
                        fig = pio.from_json(fig_json)
                        st.plotly_chart(fig, key=chart_id, use_container_width=True)
                st.markdown("---")
        
        user_question=st.chat_input("Ask a performance question: ")
        if user_question:
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            q = user_question.lower()

            if "functional scenario" in q:
                    st.subheader("Functional Scenario Data")

                    if "mmyy" in df2.columns:
                        months = [str(m).strip().lower() for m in df2["mmyy"].unique()]
                        mentioned_months = [m for m in months if m in q]  # <-- your simple logic
                        if mentioned_months:
                            filtered_data=df2["mmyy"].isin(mentioned_months)
                            filtered_data_style=df2[filtered_data].copy()
                            for col in ["total duration", "latency target"]:
                                if col in filtered_data_style.columns:
                                    filtered_data_style[col] = pd.to_numeric(filtered_data_style[col], errors="coerce")




                            if {"total duration", "latency target"}.issubset(set(filtered_data_style.columns)):
                                styled = filtered_data_style.style.apply(
                                    lambda col: np.where(
                                        filtered_data_style["total duration"] > filtered_data_style["latency target"],
                                        "color: red", ""
                                    ),
                                    axis=0,
                                    subset=["total duration"]
                                )
                                st.session_state.functional_table = styled
                                st.dataframe(styled, use_container_width=True)
                            else:
                                st.session_state.functional_table = filtered_data_style
                                st.dataframe(filtered_data_style, use_container_width=True)


                        st.session_state.qa_log.append({
                                "question": user_question,
                                "answer": "Displayed functional scenario table."
                            })


            else:


                initial_state = {
                "question": user_question,
                "summary_data": "",
                "raw_data":"", 
                "buyer_list":[],
                "buyer_data":"",
                "scenario_data":"",
                "charts":"",
                "llm_response": "",
                "messages": st.session_state.chat_history,
            }
                config = {
                "configurable": {"thread_id": st.session_state.thread_id}
            }
                final_state = graph.invoke(initial_state, config, stream_mode="values")
                ai_text = final_state["llm_response"]
                if "messages" in final_state and final_state["messages"]:
                    st.session_state.chat_history = final_state["messages"]
                else:
                    st.session_state.chat_history.append(AIMessage(content=ai_text))  
                
                st.write(user_question)
                st.write(ai_text)
                chart_jsons = final_state["charts"].charts if "charts" in final_state and final_state["charts"].charts else []
                

                # for chart_json in chart_jsons:
                #     fig = pio.from_json(chart_json)
                #     st.plotly_chart(fig,key=chart_json.id, use_container_width=True)

                # for idx, chart_json in enumerate(chart_jsons):
                #     fig = pio.from_json(chart_json)  # chart_json is already a string
                #     st.plotly_chart(fig, key=f"chart_{idx}", use_container_width=True)
                for chart_item in chart_jsons:
                    fig = pio.from_json(chart_item.fig_json)   # 👈 pass the string
                    st.plotly_chart(fig, key=chart_item.id, use_container_width=True)

                st.session_state.qa_log.append({
                "question": user_question,
                "answer": ai_text,
                "charts": [{"id": c.id, "fig_json": c.fig_json} for c in chart_jsons]
            })

