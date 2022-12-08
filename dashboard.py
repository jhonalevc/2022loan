import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(layout="wide")
from sqlalchemy import create_engine
import plotly.express as plx
import process_mining_acc_utils 




## Database Connection --- Create Dinamic Way of connecting 
engine = create_engine("postgresql://project-loan:accelirate-project-loan@20.169.221.14:5051/root")

selectbox_ = process_mining_acc_utils.selectbox.sel_box()



if selectbox_ == 'Overview':
    process_mining_acc_utils.title_centered_h1("Overveiw")
    process_mining_acc_utils.overview.container_overveiw_data(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.overview.container_overview_lineplot(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.overview.plots_distribution_three(connection=engine)

if selectbox_ == 'Process':
    process_mining_acc_utils.title_centered_h1("Process")
    content = process_mining_acc_utils.process.build_process_heu_data(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.title_centered_h3("Business Process Model")
    process_mining_acc_utils.process.build_process_BPMNs(option=content[1],connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.process.buil_variants_con_process(connection=engine,head=10)
    process_mining_acc_utils.line()
    process_mining_acc_utils.process.build_additional_maps(connection=engine,option=content[1])

if selectbox_ == "Timing":
    process_mining_acc_utils.title_centered_h1("Timing")
    process_mining_acc_utils.timing.build_timing_header(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.timing.build_timing_variants(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.timing.timing_progress_maps(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.timing.build_variant_activity(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.timing.build_variant_case_identifier(connection=engine)
    process_mining_acc_utils.line()
    process_mining_acc_utils.timing.progress_line_plot(connection=engine)

if selectbox_ == "Data":
    process_mining_acc_utils.data.data_page(connection=engine)
    





    
    


    


