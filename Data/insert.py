#!/usr/bin/python
#-*-coding: utf-8 -*-
import MySQLdb as mdb
import sys

try:
	con = mdb.connect('localhost','root','Root1234','NHIS')
	cur = con.cursor()
	sql ="create table T4_1 (HCHK_YEAR varchar(5), PERSON_ID int, YKIHO_GUBUN_CD varchar(3), HEIGHT int, WEIGHT int, WAIST int, BP_HIGH int, BP_LWST int, BLDS int,TOT_CHOLE int,TRIGLYCERIDE int,HDL_CHOLE int,LDL_CHOLE int,HMG int,GLY_CD varchar(2), OLIG_OCCU_CD varchar(2), OLIG_PH int, OLIG_PROTE_CD varchar(2), CREATININE int, SGOT_AST int,SGPT_ALT int, GAMMA_GTP int, HCHK_PMH_CD1 varchar(2), HCHK_PMH_CD2 varchar(2),HCHK_PMH_CD3 varchar(2),FMLY_LIVER_DISE_PATIEN_YN varchar(2), FMLY_HPRTS_PATIEN_YN varchar(2),FMLY_APOP_PATIEN_YN varchar(2), FMLY_HDISE_PATIEN_YN varchar(2), FMLY_DIABML_PATIEN_YN varchar(2), FMLY_CANCER_PATIEN_YN varchar(2), SMK_STAT_TYPE_RSPS_CD varchar(2), SMK_TERM_RSPS_CD varchar(2), DSQTY_RSPS_CD varchar(2), DRNK_HABIT_RSPS_CD varchar(2), TM1_DRKQTY_RSPS_CD varchar(2), EXERCI_FREQ_RSPS_CD varchar(2));"
	cur.execute(sql);
	print "making table complete"
        print "inserting 2002"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2002.txt' into table T4_1 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2003"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2003.txt' into table T4_1 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2004"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2004.txt' into table T4_1 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2005"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2005.txt' into table T4_1 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2006"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2006.txt' into table T4_1 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2007"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2007.txt' into table T4_1 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2008"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2008.txt' into table T4_1 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
	sql = "create table T4_2 (HCHK_YEAR varchar(4), PERSON_ID int, YKIHO_GUBUN_CD varchar(3), HEIGHT int, WEIGHT int, WAIST int, BP_HIGH int, BP_LWST int, BLDS int, TOT_CHOLE int,TRIGLYCERIDE int,HDL_CHOLE int,LDL_CHOLE int,HMG int, OLIG_PROTE_CD varchar(2), CREATININE int, SGOT_AST int,SGPT_ALT int,GAMMA_GTP int, HCHK_APOP_PMH_YN varchar(2), HCHK_HDISE_PMH_YN varchar(2),HCHK_HPRTS_PMH_YN varchar(2),HCHK_DIABML_PMH_YN varchar(2), HCHK_HPLPDM_PMH_YN varchar(2), HCHK_PHSS_PMH_YN varchar(2), HCHK_ETCDSE_PMH_YN varchar(2), FMLY_APOP_PATIEN_YN varchar(2), FMLY_HDISE_PATIEN_YN varchar(2), FMLY_HPRTS_PATIEN_YN varchar(2),FMLY_DIABML_PATIEN_YN varchar(2),FMLY_CANCER_PATIEN_YN varchar(2), SMK_STAT_TYPE_RSPS_CD varchar(2),PAST_SMK_TERM_RSPS_CD int, PAST_DSQTY_RSPS_CD int, CUR_SMK_TERM_RSPS_CD int,CUR_DSQTY_RSPS_CD int,DRNK_HABIT_RSPS_CD varchar(2), TM1_DRKQTY_RSPS_CD int, MOV20_WEK_FREQ_ID varchar(2), MOV30_WEK_FREQ_ID varchar(2),  WLK30_WEK_FREQ_ID varchar(2));"
	cur.execute(sql);
	print "making table complete"
        print "inserting 2009"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2009.txt' into table T4_2 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2010"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2010.txt' into table T4_2 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2011"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2011.txt' into table T4_2 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2012"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2012.txt' into table T4_2 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
	print "complete"
        print "inserting 2013"
	sql = "load data local infile '/data/nhis/12.건강검진/nhid_gj_2013.txt' into table T4_2 fields terminated by '\t' ignore 1 lines"
	cur.execute(sql);
        print "complete"
except mdb.Error, e:
	print "Error %d: %s" % (e.args[0], e.args[1])
	sys.exit(1)
