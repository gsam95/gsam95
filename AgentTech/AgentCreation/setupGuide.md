## **Step-by-Step Guide for API Setup and Model Initialization**

### **Setting Up OpenAI API**

1. **Create an Account**:
- Visit [OpenAI Platform](https://platform.openai.com/) and login/sign up
  
2. **Get API Key**:
- Go to account settings.
- Navigate to "API Keys" ([OpenAI API Keys](https://platform.openai.com/account/api-keys)).
- Click "Create new secret key" and copy the generated key (it will only be shown once).

3. **Store the API Key Securely**:
- Create a `.env` file in project directory.
- Add the following line to the file:
  ```
  OPENAI_API_KEY=your_key_here
  ```

4. **Initialize OpenAI in Code**:
from langchain_openai import ChatOpenAI

openai_llm = ChatOpenAI(
model_name="gpt-4", # or "gpt-3.5-turbo" for a less expensive option
temperature=0,
api_key=os.getenv("OPENAI_API_KEY")
)


---

### **Setting Up Groq API**

1. **Create an Account**:
- Visit [Groq Console](https://groq.com/) and sign up

2. **Get Your API Key**:
- In the Groq console, go to "API Keys."
- Generate a new API key and copy it.

3. **Store the API Key Securely**:
- Add this line to your `.env` file:
  ```
  GROQ_API_KEY=your_groq_key_here
  ```

4. **Initialize Groq in Code**:
from langchain_groq import ChatGroq

For LLaMA 3 8B model
groq_llm_llama3_8b = ChatGroq(
model_name="llama3-8b-8192",
temperature=0,
api_key=os.getenv("GROQ_API_KEY")
)

For LLaMA 3 70B model
groq_llm_llama3_70b = ChatGroq(
model_name="llama3-70b-8192",
temperature=0,
api_key=os.getenv("GROQ_API_KEY")
)


---

### **Setting Up Yahoo Finance API**

No API key is required for Yahoo Finance integration! The `yfinance` library is used to fetch stock data directly.

Example usage:
import yfinance as yf

stock = yf.Ticker("AAPL")
data = stock.history(period="1d")
print(data)


---

## **Using LLMs in LangChain Agents**

1. **Create a Selection Function**:
    This function allows you to dynamically select which LLM provider and model to use.
    ```
    def get_llm(provider="openai", model="default"):
        if provider.lower() == "openai":
            return openai_llm
        elif provider.lower() == "groq":
            if model.lower() == "llama3-70b":
                return groq_llm_llama3_70b
            else:
                return groq_llm_llama3_8b
        else:
            return openai_llm  # default fallback
    ```

2. **Initialize an Agent with Selected LLM**:
    Example initialization with Groq’s LLaMA 70B model:
    ```
    llm = get_llm("groq", "llama3-70b")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    ```

---
