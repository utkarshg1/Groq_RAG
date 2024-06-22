FROM python
WORKDIR /usr/src/app
COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["python3", "-m", "streamlit", "run", "app.py"]