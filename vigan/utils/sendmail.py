import smtplib

from email.mime.text import MIMEText


def sendmail(to, subject, message, server='smtp.osupytheas.fr', port=25, sender='arthur.vigan@lam.fr'):
    # create message and configure
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to

    # send
    s = smtplib.SMTP(server)
    s.sendmail(sender, [to], msg.as_string())
    s.quit()


