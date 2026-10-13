"""OLAP explorer: roll-up, drill-down, slice, dice and pivot over the star schema."""
import plotly.express as px
import streamlit as st

from src import olap

from . import ui

PIVOT_NOTE = {
    "Roll-up / Drill-down": "Moves up or down the **time hierarchy** (Year > Quarter > Month). Rolling up aggregates to coarser levels; drilling down reveals finer detail.",
    "Slice": "Fixes **one dimension to a single value** (e.g. Policy Type = Auto) and analyses the resulting sub-cube.",
    "Dice": "Selects a **sub-cube using several dimension filters at once** (e.g. City in {Chennai, Madurai} and Year in {2024, 2025}).",
    "Pivot": "Rotates the cube to view **two dimensions as rows and columns** for cross-tab comparison.",
}


def _chart(res, x, y, color=None, key="olap"):
    d = res.copy()
    d[x] = d[x].astype(str)
    fig = px.bar(d, x=x, y=y, color=color, barmode="group", color_discrete_sequence=[ui.NAVY, "#4C8BC2", "#9DB7D5", "#C98A00"])
    ui.show(ui.style_fig(fig, 380, f"{y} by {x}"), key=key)


def render() -> None:
    ui.page_header("OLAP Explorer", "Multidimensional analysis of the claims data warehouse (star schema)")
    ops = list(PIVOT_NOTE)
    op = st.radio("OLAP operation", ops, horizontal=True)
    ui.note(PIVOT_NOTE[op])
    measures = list(olap.MEASURES)
    dims = list(olap.DIMENSIONS)
    sql, res = "", None

    if op == "Roll-up / Drill-down":
        level = st.select_slider("Hierarchy level  (left = roll-up, right = drill-down)", options=olap.TIME_HIERARCHY, value="Year")
        c1, c2 = st.columns(2)
        measure = c1.selectbox("Measure", measures, index=1)
        split = c2.selectbox("Split by (optional)", ["None", "Policy Type", "Claim Category", "City", "Provider Type"])
        res, sql = olap.cube_query([level] + ([] if split == "None" else [split]), [measure])
        _chart(res, level, measure, None if split == "None" else split, "olap_roll")

    elif op == "Slice":
        c1, c2, c3 = st.columns(3)
        sdim = c1.selectbox("Slice dimension", dims, index=4)
        value = c2.selectbox("Slice value", olap.dimension_values(sdim))
        gdim = c3.selectbox("Group by", [d for d in dims if d != sdim], index=0)
        measure = st.selectbox("Measure", measures, index=0)
        res, sql = olap.cube_query([gdim], [measure, "Fraud Rate (%)"] if measure != "Fraud Rate (%)" else [measure],
                                   {sdim: [value]})
        _chart(res, gdim, measure, None, "olap_slice")

    elif op == "Dice":
        c1, c2, c3 = st.columns(3)
        f_policy = c1.multiselect("Policy Type", olap.dimension_values("Policy Type"))
        f_city = c2.multiselect("City", olap.dimension_values("City"))
        f_year = c3.multiselect("Year", olap.dimension_values("Year"))
        c4, c5 = st.columns(2)
        gdim = c4.selectbox("Group by", dims, index=2)
        measure = c5.selectbox("Measure", measures, index=1)
        res, sql = olap.cube_query([gdim], [measure], {"Policy Type": f_policy, "City": f_city, "Year": f_year})
        _chart(res, gdim, measure, None, "olap_dice")

    else:
        c1, c2, c3 = st.columns(3)
        rdim = c1.selectbox("Rows", dims, index=4)
        cdim = c2.selectbox("Columns", dims, index=0)
        measure = c3.selectbox("Measure", measures, index=5)
        if rdim == cdim:
            st.warning("Choose different dimensions for rows and columns.")
            return
        res, sql = olap.cube_query([rdim, cdim], [measure])
        pv = res.pivot(index=rdim, columns=cdim, values=measure).fillna(0)
        pv.columns = [str(c) for c in pv.columns]
        heat = px.imshow(pv, text_auto=".1f" if "Rate" in measure or "Score" in measure else ".0f", aspect="auto",
                         color_continuous_scale=["#FFFFFF", "#9DB7D5", "#1F3A5F"])
        ui.show(ui.style_fig(heat, 420, f"{measure}: {rdim} x {cdim}"), key="olap_pivot")
        res = pv.reset_index()

    ui.section("Result Table")
    if res is None or res.empty:
        st.info("The selected cube cell range contains no data.")
    else:
        ui.html_table(res, max_height=380)
        st.download_button("Download result (CSV)", res.to_csv(index=False).encode("utf-8"),
                           file_name="olap_result.csv", mime="text/csv")
    with st.expander("View generated SQL"):
        st.code(sql, language="sql")
