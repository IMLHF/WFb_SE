import smtplib
import os
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Watch GPU situation
YesGPU = False
NoGPU = 1
usedMemory=[]
i = 0
threshold = 2000 # less than 2000MB regarded as available
totalMemory = os.popen('nvidia-smi --query-gpu=memory.total --format=csv,noheader').read().split()[0]

while YesGPU == False and i != NoGPU:
    time.sleep(10)
    r = os.popen("nvidia-smi")
    text = r.read()
    usedMemory.clear()
    for i in range(0,NoGPU): # I have 6 GPUs
        index = text.find('/ '+totalMemory)  # Note: SPACE here, totalMemory is got automatically.
        usedMemory.append(int(text[int(index)-9:int(index)-4]))
        print(usedMemory[i])
        text = text[index+10:]
        if int(usedMemory[i]) < threshold:  # less than $threshold regarded as available
            YesGPU = True
    r.close()
    print(usedMemory)

# Send email
MemorySize = min(usedMemory) # find the most available GPU No.
MemoryNo = usedMemory.index(min(usedMemory))
print(MemorySize,MemoryNo)

# sender = 'sender@mail.com' # Sender email address
# receiver = list()
# receiver.append('receiver@mail.com') # Receiver email address
# copyReceive = list()
# copyReceive.append(sender)
# username = 'sender@mail.com' # log in to the email sever
# password = '********'# password
mailall=MIMEMultipart()
mailall['Subject'] = 'GPU' + str(MemoryNo) + 'Avaliable' + str(MemorySize) + 'MB' # Subject of the email
mailall['From'] = sender
mailall['To'] = ';'.join(receiver)
mailall['CC'] = ';'.join(copyReceive)
mailcontent = 'None'
mailall.attach(MIMEText(mailcontent, 'plain', 'utf-8'))
mailAttach = 'None'
contype = 'application/octet-stream'
maintype, subtype = contype.split('/', 1)
# filename = '????.txt'#????????
# attfile = MIMEBase(maintype, subtype)
# attfile.set_payload(open(filename, 'rb').read())
# attfile.add_header('Content-Disposition', 'attachment',filename=('utf-8', '', filename))
# mailall.attach(attfile)
fullmailtext = mailall.as_string()
smtp = smtplib.SMTP_SSL(host='smtp.yeah.net')
print("Connecting")
smtp.connect("smtp.yeah.net")
print("Logining")
smtp.login(username, password)
print("Sending")
smtp.sendmail(sender, receiver, fullmailtext)
smtp.quit()
print("GPU is free now! Finished")
