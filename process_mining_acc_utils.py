import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import streamlit as st
import streamlit_nested_layout
import plotly.express as plx
import datetime
from PIL import Image
from io import BytesIO
import json
import random


def title_centered_h3(str_):
    title = st.markdown("""<h3 style='text-align: center'>""" + str(str_) +"""</h3>""",unsafe_allow_html =True)
    return title
def title_centered_h4(str_):
    title = st.markdown("""<h4 style='text-align: center'>""" + str(str_) +"""</h4>""",unsafe_allow_html =True)
    return title
def title_centered_h1(str_):
    title = st.markdown("""<h1 style='text-align: center'>""" + str(str_) +"""</h1>""",unsafe_allow_html =True)
    return title
def cent_text(str_):
    title = st.markdown("""<p style='text-align: center'>""" + str(str_) +"""</p>""",unsafe_allow_html =True)
    return title
def line():
    return st.markdown("""<hr>""",unsafe_allow_html=True)
def space():
    return st.markdown("""<br>""",unsafe_allow_html =True)


class selectbox:
    def sel_box():
        selectbox_ = st.sidebar.selectbox("Available Options for you to explore",('Overview','Timing','Process','Data'))
        with st.sidebar:
            space()
            st.info("If assistance is required contact platform admin")
        return selectbox_

class overveiw_header_download():
    def __init__(overveiw_header_download,script,connection):
        overveiw_header_download.script = script
        overveiw_header_download.connection = connection
    
    def download_data_sql(argument):
        table = pd.read_sql(sql = argument.script , con= argument.connection)
        return table

class header_summary_plots():
    def __init__(header_summary_plots):
        pass
    def test(ahhh, ahh2 ):
        overveiw_header_build.download_data_sql()

class build_header_streamlit():
    def __init__(build_header_streamlit, table):
        build_header_streamlit.table = table

    def container_with_header(arg):
        columns = arg.table.columns.to_list()
        container_with_header =  st.container()
        with container_with_header:
            f1,f2,f3,f4 = st.columns(4)
            list_d = [f1,f2,f3,f4]
            for column_,column_name in zip(list_d,columns):
                with column_:
                    title_centered_h3(column_name)
                    cent_text(arg.table.loc[0,column_name])
            
            return container_with_header

class overveiw_header_build:
    def __init__(overheaderinfo,cases,events,activities,variants):
        overheaderinfo.cases = cases
        overheaderinfo.events = events
        overheaderinfo.activities = activities
        overheaderinfo.variants = variants
    def myfunc(argument):
        print('Hello, This are the activities and variants:  ' + argument.cases + ' '  + ' & ' +  argument.variants )

class overview:

    def container_overveiw_data(schema= 'public',table= 'overview_header',connection=""):
        script = """SELECT * FROM """ + schema + """."""+ table
        dataframe = pd.read_sql(sql= script,con=connection)
        year_selected = st.selectbox('Year', dataframe['year'].unique().tolist())
        dataframe2 = dataframe[dataframe['year'] == year_selected]
        container_overveiw = st.container()
        with container_overveiw:    
            f1,f2,f3,f4 = st.columns(4)
            with f1:
                title_centered_h3('Cases')
                cent_text(str(dataframe2['cases'].to_list()[0]))
            with f2:
                title_centered_h3('Events')
                cent_text(str(dataframe2['events'].to_list()[0]))
            with f3:
                title_centered_h3('Activities')
                cent_text(str(dataframe2['activities'].to_list()[0]))
            with f4:
                title_centered_h3('Variants')
                cent_text(str(dataframe2['variants'].to_list()[0]))        
        return container_overveiw



    def container_overview_lineplot(schema ="public", table = "eventlog_df",connection="",cont_width =True):
        container_lineplot_1 = st.container()
        with container_lineplot_1:
            fg1, fg2 = st.columns(2)
            with fg1:
                global selection_fg1
                selection_fg1 = st.radio('Year - Month',['Year','month'])
            if selection_fg1 == 'Year':
                script = """SELECT "time:timestamp" ::date as date, count("case:concept:name") FROM """  + schema +""".""" + table + """  GROUP BY date  order by date """
                dataframe = pd.read_sql(con= connection,sql= script)
                dataframe['date'] = pd.to_datetime(dataframe['date'])
                with fg2:
                    global selector_fg2_1
                    selector_fg2_1 = st.selectbox('Select Year', dataframe['date'].dt.year.drop_duplicates().to_list() + ['total'])
                if selector_fg2_1 == 'total':
                    plot_1 = plx.area(data_frame = dataframe, x='date',y='count')
                    st.plotly_chart(plot_1,use_container_width=cont_width)
                else:
                    df = dataframe[dataframe['date'].dt.year == selector_fg2_1]
                    plot_1 = plx.area(data_frame = df, x='date',y='count')
                    st.plotly_chart(plot_1,use_container_width=cont_width)
            else:
                with fg2:
                    st.error('Monthly Analysis')
                script = """SELECT TO_CHAR("time:timestamp" ::date, 'Month') ,COUNT("case:concept:name") FROM """ +  schema + """.""" + table + """ GROUP BY TO_CHAR("time:timestamp" ::date, 'Month');"""
                dataframe = pd.read_sql(con= connection,sql= script)
                dataframe.columns = ['month','count']
                plot_1 = plx.area(data_frame = dataframe, x='month',y='count')
                st.plotly_chart(plot_1,use_container_width=cont_width)
                
            
    def plots_distribution_three(schema = "public",table ="variants_info",connection=""):
        d_container = st.container()
        with d_container:
            yu1,yu2,yu3 = st.columns(3)
            with yu1:
                title_centered_h3("Variants")
                script = """SELECT "Variant_Name" , ROUND(percentage::numeric,2) as percentage FROM """  + schema  + """.""" + table
                dataframe = pd.read_sql(sql = script,con= connection)
                plot = plx.bar(data_frame = dataframe.head(10), x = 'Variant_Name', y = "percentage")
                st.plotly_chart(plot,use_conatiner_width = True)
            with yu2:
                title_centered_h3('Events per case')
                script = """SELECT "count" as "#activities",COUNT("dummy") as count, (COUNT("dummy") / sum(COUNT("dummy")) OVER () * 100) AS PERCENTAGE 
                        FROM (SELECT "case:concept:name" as "case", 1 as dummy, count("time:timestamp") 
                        FROM """ +  schema + """.""" + """eventlog_df GROUP BY "case") Q1 GROUP BY "#activities" ORDER BY percentage DESC"""             
                dataframe = pd.read_sql(sql = script,con= connection)
                plot = plx.bar(data_frame = dataframe.head(10), y = '#activities', x = "percentage", orientation = "h")
                st.plotly_chart(plot,use_conatiner_width = True)
            with yu3:
                title_centered_h3("Activities")
                script = """ SELECT "Variant_Name", "count", ("count"/sum("count") over() * 100) as percentage
                        FROM
                        (SELECT "Variant_Name", count("Case") FROM """ +  schema + """.""" + """variants_info_percase
                        GROUP BY "Variant_Name") as q2
                        ORDER BY percentage DESC  """
                dataframe = pd.read_sql(sql = script,con= connection)
                plot = plx.bar(data_frame = dataframe.head(10), x = 'Variant_Name', y = "percentage")
                st.plotly_chart(plot,use_conatiner_width = True)
        return d_container

class process:

    def build_process_heu_data(connection= "", schema = "public",table = "process_heuristic_header_data"):

        images_dataframes = pd.read_sql(con = connection, sql = """SELECT images ::bytea, start_activities, end_activities, complexity FROM """ + schema + """.""" + table)
        options = images_dataframes['complexity'].sort_values(ascending=True).to_list()
        _container_ = st.container()
        
        with _container_:
            slider_complexity = st.select_slider(label = 'Slide',options = options)
            df_work = images_dataframes[images_dataframes['complexity'] == slider_complexity]            
            
            col1_, col2_ = st.columns(2)

            with col1_:
                global image_bin
                image_bin = Image.open(BytesIO(df_work.iloc[0,0]))
                st.image(image_bin)
            
            with col2_:
                global df_ending_act
                title_centered_h3("Starting Activities")
                df_start_act = pd.DataFrame(json.loads(df_work.iloc[0,1]))
                st.table(df_start_act)
                space()
                title_centered_h3("Ending Activitiess")
                df_ending_act = pd.DataFrame(json.loads(df_work.iloc[0,2]))
                st.table(df_ending_act)
                space()
                title_centered_h3("Download Net")
                space()
                cc1, cc2, cc3,cc4,cc5 = st.columns(5)
                with cc3:
                    st.download_button("Download", data = BytesIO(df_work.iloc[0,0]),mime= 'image/jpeg')
            
            line()

            with st.expander("Additional Info"):
                title_centered_h3("Percentage of Ending Activities")
                plot_ending = plx.bar(x = df_ending_act.columns, y = df_ending_act.loc['percentage',:]  )
                st.plotly_chart(plot_ending,use_container_width =True)

        return _container_, slider_complexity



    def build_process_BPMNs(option =None, connection =None, schema= "public", table= "process_bpmn_images"):
        images_dataframes = pd.read_sql(con = connection, sql = """SELECT image ::bytea,complexity FROM """ + schema + """.""" + table)
        _d_container_ = st.container()
        with _d_container_:
            byte_im = images_dataframes[images_dataframes['complexity'] == option]['image'].to_list()[0]
            image_bin = Image.open(BytesIO(byte_im))
            st.image(image_bin,use_column_width =True)
        return _d_container_
   
    def buil_variants_con_process(connection = "", schema = "public",table = "variants_info", head = 100):


        script = """SELECT "Variant","case_len","percentage","Variant_Name" FROM """ + schema + """.""" + table
        df = pd.read_sql(sql=script, con= connection)
        df['percentage'] = pd.to_numeric(df['percentage'])
        df['case_len'] = pd.to_numeric(df['case_len'])
        df['Variant'] = df['Variant'].astype(str)
        df['Variant'] = df['Variant'].str.replace("""'""","")
        container_l = st.container()
        with container_l:
            title_centered_h3("Variants")
            l1,l2,l3 = st.columns([2,1,1])
            with l1:
                title_centered_h4("Select Variant")
                global selector_variant
                selector_variant = st.selectbox("select Variant",df['Variant_Name'].to_list())        
            
            df_trimmed = df[df['Variant_Name'] == selector_variant]
            with l2:
                title_centered_h4("Length of the Trace")
                st.info("Number of elements: " + str(df_trimmed['Variant'].to_list()[0].count(",") + 1))
                st.info("Percentage of occurance: " + str(df_trimmed['percentage'].round(2).to_list()[0]) + " %")
            with l3:
                title_centered_h4("Elements of the Trace")
                st.markdown(df_trimmed['Variant'].to_list()[0])

            s1,s2 = st.columns(2)
            with s1:
                global sel_opt
                sel_opt = st.radio(label ="selection",label_visibility= "hidden",options = ['length of variance', 'Ocurrance of Variance'], horizontal = True)

            with  s2:
                global number
                number = st.number_input('Insert a number ov Variants to view', value = 150)

            if sel_opt == 'length of variance':
                plot_len = plx.bar(data_frame = df.head(number), x = 'Variant_Name', y = 'case_len')
                st.plotly_chart(plot_len, use_container_width = True)
            else:
                plot_len = plx.bar(data_frame = df.head(number), x = 'Variant_Name', y = 'percentage')
                st.plotly_chart(plot_len, use_container_width = True)


        return container_l

    def build_additional_maps(connection ="",schema="public",table="additional_maps",option=None):

        script = """SELECT petri_net_inductive::bytea, petri_net_alpha::bytea,petri_net_alpha_plus::bytea,complexity FROM """ +  schema + """.""" + table
        dataframe = pd.read_sql(con=connection,sql = script)

        e_cont = st.container()
        with e_cont:

            title_centered_h3("Aditional Process Maps")

            d1,d2 = st.columns ([5,1])
            with d1:
                big_image_byte = dataframe[dataframe['complexity'] == option]['petri_net_inductive'].to_list()[0]
                image_ = Image.open(BytesIO(big_image_byte))
                st.image(image_,use_column_width=True)

            with d2:
                st.download_button("Download Petri Net Inductive", data = BytesIO(dataframe[dataframe['complexity'] == option].iloc[0,0]),mime= 'image/jpeg')
                st.download_button("Download Petri Net Alpha Miner ", data = BytesIO(dataframe[dataframe['complexity'] == option].iloc[0,1]),mime= 'image/jpeg')
                st.download_button("Download Petri Net Alpha + Miner ", data = BytesIO(dataframe[dataframe['complexity'] == option].iloc[0,2]),mime= 'image/jpeg')

class timing:

    def build_timing_header(connection ="",table="eventlog_df",schema ="public"):
        SCRIPT = """
        SELECT 
            (SELECT MAX("time:timestamp") FROM """ + schema + """.""" + table + """) AS "MAX",
            (SELECT MIN("time:timestamp") FROM """ + schema + """.""" + table + """) AS "MIN"
        """

        df_simple = pd.read_sql(sql = SCRIPT, con= connection )
        f_conta = st.container()
        with f_conta:

            d1,d2,d3,d4,d5,d6 = st.columns(6)
            with d1:
                title_centered_h3("Initial Time")
                cent_text(df_simple["MIN"].to_list()[0])
            with d6:
                title_centered_h3("Ending Time")
                cent_text(df_simple["MAX"].to_list()[0])
            
            for i in [d2,d3,d4,d5]:
                with i:
                    line()
        
        diff = df_simple.iloc[0,0] - df_simple.iloc[0,1]
        days = diff.days
        hours,remainder = divmod(diff.seconds,3600)
        minutes, seconds = divmod(remainder,60)
        
        time_Data = [days,hours,minutes,seconds]
        time_Data_str = ['days','hours','minutes','seconds']
        with st.expander("Time elapsed Details"):

            fd1,fd2,fd3,fd4 = st.columns(4)

            for a,b,c in zip(time_Data,time_Data_str,[fd1,fd2,fd3,fd4]):
                with c:
                    title_centered_h4(b)
                    cent_text(str(a))

    def build_timing_variants(connection="",table ="variants_info_percase",schema="public",table_2 = "eventlog_df"):


        SCRPT_1 = """ SELECT "Case" FROM """  + schema + """.""" + table
        cases_list = pd.read_sql(con=connection,sql=SCRPT_1)

        we1,we2 = st.columns([2,1])

        with we1:
            global case_selector
            global df2_
            case_selector = st.selectbox(label = "Select Case", options =  cases_list['Case'].to_list())
            idx_list = (cases_list.index.to_list())
            rand_list = []
            for i in np.arange(5):
                rand_list.append(random.randrange(len(idx_list)))

            elems = []
            for i in rand_list:
                elems.append(cases_list.loc[i,'Case'])

            elems_tuple_str = str(tuple(elems + [case_selector]))
            SCRIPT2 = """
            SELECT "case:concept:name" as Case, ROUND(EXTRACT(EPOCH FROM (max - min)) / 60,2) as Interval FROM (
            SELECT "case:concept:name" , MIN("time:timestamp") AS MIN , MAX("time:timestamp") AS MAX
            FROM """ +  schema + """.""" + table_2 + """  
            WHERE "case:concept:name" IN """ +  elems_tuple_str + """
            GROUP BY "case:concept:name"
            ) AS Q
            """
            df2_ = pd.read_sql(sql=SCRIPT2, con=connection)
            bar_plot_len_case = plx.bar(data_frame = df2_, x = 'case', y = 'interval')
            st.plotly_chart(bar_plot_len_case,use_container_width = True)
        
        with we2:

            gf1, gf2, gf3 = st.columns(3)

            with gf1:
                title_centered_h4("Variant")
                script3 = """SELECT "Variant_Name" FROM """ + schema + """."""  +table + """ WHERE "Case" = """ + str("'" + case_selector + "'" )
                d_df =  pd.read_sql(con=connection,sql = script3)
                space()
                cent_text(d_df['Variant_Name'].to_list()[0])
            with gf2:
                title_centered_h4("Case")
                space()
                cent_text(str(case_selector))
            with gf3:
                title_centered_h4("Time Elapsed in Hours")
                cent_text(str(df2_[df2_['case'] == case_selector]['interval'].to_list()[0]))
            line()

            script4 = """ SELECT "time:timestamp" AS TIME , "concept:name" AS ACTIVITY ,"case:concept:name" AS CASE  FROM """ + schema + """.""" + table_2 +  """ WHERE "case:concept:name" = """ + str("'"+case_selector + "'")
            df__ = pd.read_sql(con=connection,sql=script4)
            df__['time'] = pd.to_datetime(df__['time'])
            df__['shifted'] = df__['time'].shift(-1)
            df__['interval'] = df__['shifted'] - df__['time']  
            df__['interval'] = df__['interval'].astype('timedelta64[m]')
            df__['interval'] = df__['interval'].fillna(0).astype(float)
            st.table(df__[['time','activity','interval']])

    def build_variant_activity(connection="",schema="public", table1 = "variants_info",table2 ="variants_info_percase",table3 = "eventlog_df"):

        container_elem = st.container()
        with container_elem:    
            c_col1, c_col2 = st.columns([1,2])
            with c_col1:
                global act_vari_selector
                act_vari_selector = st.radio(label = "Seelct", label_visibility ="hidden",options = ["Variant","Activity"],horizontal =True)

            if act_vari_selector == "Variant":
                with c_col2:
                    global variant_Selector, df_1
                    script = """SELECT "Variant_Name" FROM  """ + schema + """.""" + table1
                    df_1 = pd.read_sql(con= connection,sql= script)
                    variant_Selector = st.selectbox(label = "select Variant",label_visibility ="hidden",options = df_1['Variant_Name'].to_list())

                w_col1_, w_col2_ = st.columns([1,2])
                with w_col1_:
                    random_vars = []
                    for i in np.arange(5):
                        random_vars.append("Variant_"+str(random.randrange(len(df_1))))
                    script2 =  """
                        SELECT 
                            "Variant_Name", 
                            AVG(EXTRACT(EPOCH FROM max-min) /60) as "minutes"
                        FROM (
                            SELECT 
                                t1."Variant_Name" ,
                                t1."Case" as "case", 
                                min(t2."time:timestamp") as min, 
                                max(t2."time:timestamp") as max 
                            FROM 
                                """ + schema + """.""" + table2 + """ t1
                            LEFT JOIN
                                """ +  schema + """.""" + table3 + """ t2
                            ON 
                                t1."Case" = t2."case:concept:name"
                            WHERE 
                                t1."Variant_Name" IN """ +  str(tuple(random_vars + [variant_Selector])) + """ 
                            group by  
                            t1."Variant_Name",t1."Case") AS F_TABLE
                        GROUP BY 
                            "Variant_Name" """
                    df_variants_interval = pd.read_sql(con= connection, sql=script2)
                    plot_variants_interval = plx.bar(orientation = "h", data_frame= df_variants_interval, y = 'Variant_Name', x = "minutes")
                    st.plotly_chart(plot_variants_interval,use_container_width=True)
                
                with w_col2_:
                    script3 = """
                    SELECT "concept", MIN(seconds_passed) AS MIN,MAX(seconds_passed) AS MAX, AVG(seconds_passed) as AVG,
                    PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY SECONDS_PASSED) as MEDIAN,  PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY SECONDS_PASSED) as Q75, PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY SECONDS_PASSED) as Q25,STDDEV(SECONDS_PASSED) AS STD 
                    FROM
                    (
                        SELECT "Case","concept", ROUND(EXTRACT(EPOCH FROM lead-time),4) as SECONDS_PASSED
                        FROM(
                            SELECT "Case","concept", "time", lead("time",1) over(PARTITION by "Case" ORDER BY "time")
                            FROM
                            (
                                    SELECT t1."Variant_Name",t1."Case",t2."concept:name" as "concept" ,t2."time:timestamp" as time
                                    FROM """ + schema + """.""" + table2 + """ t1
                                    LEFT JOIN """ + schema + """.""" +  table3 + """ t2
                                    ON t1."Case" = t2."case:concept:name"
                                    WHERE "Variant_Name" = """ + "'"+ variant_Selector + "'" + """ 
                            ) TST
                        ) TST2
                    )TST3
                    GROUP BY "concept"
                    """
                    df_concept_summary = pd.read_sql(con = connection,sql = script3).fillna(0)
                    space()
                    space()
                    st.table(df_concept_summary)

            else:
                w_col1_, w_col2_ = st.columns([1,2])
                with c_col2:
                    global activity_select
                    script4 = """ SELECT DISTINCT("concept:name") FROM """ + schema + """.""" + table3
                    activity_select = st.selectbox(label = "Select Activity", label_visibility = "hidden", options = pd.read_sql(con= connection,sql = script4)['concept:name'])
                
                space()

                with w_col1_:
                    script5 = """
                        SELECT 
                            "concept",
                            count("seconds_passed") as count, 
                            min("seconds_passed") as min, 
                            max("seconds_passed") as max,
                            avg("seconds_passed") as avg,
                            STDDEV("seconds_passed") as STD,
                            PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY "seconds_passed") as MEDIAN,
                            PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY "seconds_passed") as q75,
                            PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY "seconds_passed") as q25
                            
                        FROM
                            ( 
                            SELECT "concept", ROUND(EXTRACT(EPOCH FROM lead-time),4) as SECONDS_PASSED
                            FROM
                                (
                                SELECT "concept","time",lead("time",1) OVER(PARTITION BY "Case" ORDER BY "time")
                                FROM
                                        (
                                        SELECT t1."Case",t2."concept:name" as "concept" ,t2."time:timestamp" as time
                                        FROM """ + schema + """.""" + table2 + """ t1
                                        LEFT JOIN """ +  schema + """.""" + table3 + """ t2
                                        ON t1."Case" = t2."case:concept:name"
                                        ) TST
                                ) TST2
                            )TST3
                        WHERE "concept" =  """ + "'" + activity_select + "'" + """
                        GROUP BY "concept" """

                    df_activity = pd.read_sql(con = connection, sql = script5)
                    df_activity = df_activity.T
                    df_activity.columns = df_activity.iloc[0,:]
                    df_activity = df_activity.drop("concept")
                    st.table(df_activity)

                with w_col2_:

                    with st.expander("Box Plot") :
                        script6 = """
                                SELECT "concept", ROUND(EXTRACT(EPOCH FROM lead-time),4) as interval
                                FROM
                                    (
                                    SELECT "concept","time",lead("time",1) OVER(PARTITION BY "Case" ORDER BY "time")
                                    FROM
                                            (
                                            SELECT t1."Case",t2."concept:name" as "concept" ,t2."time:timestamp" as time
                                            FROM """ +  schema + """.""" + table2 + """ t1
                                            LEFT JOIN """ + schema +  """.""" + table3 + """ t2
                                            ON t1."Case" = t2."case:concept:name"
                                            ) TST
                                    ) TST2 
                                WHERE concept = """ + "'" + activity_select +"'"
                        df_boxplot = pd.read_sql(con= connection, sql= script6)
                        boxplot = plx.box(data_frame = df_boxplot, x = 'interval')
                        st.plotly_chart(boxplot,use_container_width = True)
                                
                    with st.expander("Scatter  Plot") :
                        script7 = """
                                SELECT "concept", ROUND(EXTRACT(EPOCH FROM lead-time),4) as interval,"time"
                                FROM
                                    (
                                    SELECT "concept","time",lead("time",1) OVER(PARTITION BY "Case" ORDER BY "time")
                                    FROM
                                            (
                                            SELECT t1."Case",t2."concept:name" as "concept" ,t2."time:timestamp" as time
                                            FROM """ +  schema + """.""" + table2 + """ t1
                                            LEFT JOIN """ + schema +  """.""" + table3 + """ t2
                                            ON t1."Case" = t2."case:concept:name"
                                            ) TST
                                    ) TST2 
                                WHERE concept = """ + "'" + activity_select +"'"
                        df_scatterplot = pd.read_sql(con= connection, sql= script7)
                        scatterplot = plx.scatter(data_frame = df_scatterplot, x = 'interval', y = "time")
                        st.plotly_chart(scatterplot,use_container_width = True)
                        
                    with st.expander("Line Plot") :
                        script8 = """  SELECT "date", count("concept")
                        FROM (
                            SELECT "concept", ROUND(EXTRACT(EPOCH FROM lead-time),4) as SECONDS_PASSED,DATE("time")
                            FROM
                                (
                                SELECT "concept","time",lead("time",1) OVER(PARTITION BY "Case" ORDER BY "time")
                                FROM
                                        (
                                        SELECT t1."Case",t2."concept:name" as "concept" ,t2."time:timestamp" as time
                                        FROM """ + schema + """.""" + table2 + """ t1
                                        LEFT JOIN """ + schema + """.""" + table3 + """ t2
                                        ON t1."Case" = t2."case:concept:name"
                                        ) TST
                                ) TST2 
                            WHERE concept = """ + "'" + activity_select +"'" + """
                        ) TST3
                        group by "date" """

                        df_lineplot = pd.read_sql(con= connection,sql= script8)
                        lineplot_ = plx.area(data_frame= df_lineplot, x = "date", y = "count")
                        st.plotly_chart(lineplot_, use_container_width= True)

        return container_elem

    def build_variant_case_identifier(connection="",schema='public',table1 ="variants_info",table2 ="variants_info_percase"):

        xc_container= st.container()
        with xc_container:
            with st.expander("Case & Variant"):
                xc_col1, xc_col2 = st.columns(2)
                with xc_col1:
                    title_centered_h4("Find case per Variant")
                    script_1 = """SELECT "Variant_Name" FROM  """ + schema + """.""" + table1
                    variants_df = pd.read_sql(con= connection,sql= script_1)
                    variant_Selector_ = st.selectbox(label = "select Variant __",label_visibility ="hidden",options = variants_df['Variant_Name'].to_list(),index = 87)
                    script_2 = """SELECT "Cases" FROM  """ + schema + "." + table1 + """ WHERE  "Variant_Name" =  """ + "'" + variant_Selector_ + "'"
                    data_Df = pd.read_sql(con = connection, sql = script_2)
                    cent_text(str(data_Df.iloc[0,0]))
                with xc_col2:
                    title_centered_h4("Find Variant per Case")
                    script_3 = """SELECT "Case" FROM  """ + schema + """.""" + table2
                    cases_df = pd.read_sql(con= connection,sql = script_3 )
                    case_selector_ = st.selectbox(label = "select Variant _1_",label_visibility ="hidden",options = cases_df['Case'].to_list(),index = 87)
                    script_4 = """ SELECT "Variant_Name" FROM  """ +schema + "." + table2 + """ WHERE "Case" = """ + "'" + case_selector_ + "'"
                    cases_data_df = pd.read_sql(con= connection,sql = script_4)
                    cent_text(str(cases_data_df.iloc[0,0]))
        return xc_container

    def timing_progress_maps(connection = "", schema = "public",table1 = "timing_visual"):

        container_info = st.container()
        with container_info:
            title_centered_h3("Progress Analysis")
            script1 = """SELECT image ::bytea,complexity,dfs_time,activity_df FROM """ + schema + "." + table1
            df_complex = pd.read_sql(con= connection,sql = script1)
            slider_complex = st.select_slider(label = 'Slide complexity', options = df_complex["complexity"].to_list())
            df_work = df_complex[df_complex['complexity'] == slider_complex]
            zz_col1,zz_col2 = st.columns([2,1])

            with zz_col1:
                image_bin = Image.open(BytesIO(df_work.iloc[0,0]))
                st.image(image_bin)
            with zz_col2:
                df_time_header =  pd.DataFrame(json.loads(df_work.iloc[0,2]))
                xd1,xd2,xd3 = st.columns(3)
                with xd1:
                    title_centered_h4("Days")
                    cent_text(df_time_header.iloc[0,2])
                with xd2:
                    title_centered_h4("Hours")
                    cent_text(df_time_header.iloc[0,3])
                with xd3:
                    title_centered_h4("Minutes")
                    cent_text(df_time_header.iloc[0,4])
                line()
                st.table(pd.DataFrame(json.loads(df_work.iloc[0,-1])))





                




        return container_info
        
    def progress_line_plot(connection = "",schema="public",table1 = "eventlog_df"):
        b_container = st.container()
        with b_container:
            string_1= """SELECT DISTINCT  "case:concept:name" as Case FROM """ + schema + "." + table1  
            cases_list = pd.read_sql(con= connection, sql = string_1)["case"].to_list()
            b_col1,b_col2 = st.columns([1,2])
            with b_col1:
                global a_selector,b_selector,c_selector,d_selector,e_selector
                a_selector = st.selectbox(label = "Case 1",  options = cases_list, index = 1555)
                b_selector = st.selectbox(label = "Case 2",  options = cases_list, index = 765)
                c_selector = st.selectbox(label = "Case 3",  options = cases_list, index = 67)
                d_selector = st.selectbox(label = "Case 4",  options = cases_list, index = 89)
                e_selector = st.selectbox(label = "Case 5",  options = cases_list, index = 891)

            str_ = "(" + "'"+ a_selector +"'" + ",""'"+ b_selector +"'" + ",""'"+ c_selector +"'" + ",""'"+ d_selector +"'" + ",""'"+ e_selector +"'" +  ")"
            string_plot = """SELECT "concept:name" AS concept, "time:timestamp" as time, "case:concept:name" as case  FROM """ + schema +"." +table1 + """ WHERE  "case:concept:name" IN """ + str_
            df_plot = pd.read_sql(con=connection,sql= string_plot)
            with b_col2:
                st.plotly_chart(plx.line(df_plot,x = 'concept', y = "time", markers= True, color= "case"),use_container_width = True)
            
        return b_container

class data:

    def data_page(connection= "", table_1 = 'eventlog_df',table_2= "variants_info_percase", table_3= "variants_info" ,schema="public" ):

        title_centered_h1("Data")
        cent_text("Please reveiw a subset of the Data")

        xc_container = st.container()
        with xc_container:
            script1 = """SELECT * FROM """ + schema + "." + table_1 + """  WHERE  "case:concept:name" IN (SELECT DISTINCT("case:concept:name") FROM """ + schema+"."+table_1 + """ limit 1 ) """
            st.table(pd.read_sql(con= connection, sql = script1))
            line()

            sd_col1, sd_col2 = st.columns(2)

            with sd_col1:
                script2 = """SELECT * FROM """ + schema + "." + table_2 + """ limit 10"""
                st.table(pd.read_sql(con= connection, sql = script2))

        return xc_container
        
        




