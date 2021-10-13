import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart #email內容載體
from email.mime.text import MIMEText #用於製作文字內文
from email.mime.base import MIMEBase #用於承載附檔
from email import encoders #用於附檔編碼
import datetime
import ssl
#寄件者使用的Gmail帳戶資訊
gmail_user = '107000129@office365.nthu.edu.tw'
gmail_password = ''
from_address = gmail_user
  
#設定信件內容與收件人資訊
df_score = pd.read_csv('./exam1.csv')  
df_score = df_score.dropna(axis=0,how='all')  
for index in range(len(df_score)):
    to_address = df_score["email"][index]
    # 標題
    Subject = "嵌入式 Exam#1 總成績"
    # 內容
    contents = "\
    {} 同學好, \n \
    嵌入式 Exam#1 總成績 : {}\n \
    \n \
    扣分1(5分) : -{}\n \
    扣分2(-) : -{}\n \
    扣分3(10分) : -{}\n \
    扣分4(5分) : -{}\n \
    \n \
    Part1.(20%) : -{}\n \
    Part2.(20%) : -{}\n \
    Part3.(20%) : -{}\n \
    Part4.(20%) : -{}\n \
    Part5.(20%) : -{}\n \
    \n \
    評分標準已公告在Teams上 \n \
    如有任何問題請在5/7 23:59前與助教聯繫 \n \
    謝謝\n \
    TA 古皓丞 \n \
    \n \
    Dear {} \n \
    You got {} on 'Embedded System Lab Exam#1' \n \
    \n \
    Deduct1(5分) : -{}\n \
    Deduct2(-) : -{}\n \
    Deduct3(10分) : -{}\n \
    Deduct4(5分) : -{}\n \
    \n \
    Part1.(20%) : -{}\n \
    Part2.(20%) : -{}\n \
    Part3.(20%) : -{}\n \
    Part4.(20%) : -{}\n \
    Part5.(20%) : -{}\n \
    \n \
    Please contact TA before 5/7 23:59 if you have any question about your score. \n \
    Thank you \n \
    \n \
    TA 古皓丞 \n ".format(df_score["姓名"][index],df_score["分數"][index],df_score["扣分1(5分)"][index],df_score["扣分2(-)"][index],df_score["扣分3(10分)"][index]
    ,df_score["扣分4(5分)"][index],df_score["Part 1.(20%)"],df_score["Part 2.(20%)"],df_score["Part 3.(20%)"],df_score["Part 4.(20%)"],df_score["Part 5.(20%)"]
    ,[index],df_score["姓名"][index],df_score["分數"][index],df_score["扣分1(5分)"][index],df_score["扣分2(-)"][index],df_score["扣分3(10分)"][index]
    ,df_score["扣分4(5分)"][index],df_score["Part 1.(20%)"],df_score["Part 2.(20%)"],df_score["Part 3.(20%)"],df_score["Part 4.(20%)"],df_score["Part 5.(20%)"])

 
    # #開始組合信件內容
    mail = MIMEMultipart()
    mail['From'] = from_address
    mail['To'] = to_address
    mail['Subject'] = Subject
    # #將信件內文加到email中
    mail.attach(MIMEText(contents))     
    

    # # 設定smtp伺服器並寄發信件    
    smtpserver = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    smtpserver.ehlo()
    smtpserver.login(gmail_user, gmail_password)
    smtpserver.sendmail(from_address, to_address, mail.as_string())
    smtpserver.quit()

# Reference & Special thanks to:
# https://lcycblog.wordpress.com/2018/09/10/python-gmail-%E5%AF%84%E4%BF%A1-with-%E9%99%84%E5%8A%A0%E5%A4%9A%E5%80%8B%E6%AA%94%E6%A1%88/

