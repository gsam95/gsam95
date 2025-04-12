# AI Financial Portfolio Rebalancer

This project is an AI-powered financial portfolio rebalancer that uses LangChain's agent framework to analyze stock portfolios, fetch real-time market data, and provide actionable recommendations for rebalancing. The tool integrates OpenAI and Groq LLMs, along with Yahoo Finance data, to deliver accurate and efficient financial insights.

---

## **Introduction**

This project showcases my expertise in developing an AI-powered financial portfolio rebalancer using LangChain's agent framework. Through this mini-project, I honed several key skills:

*   **Integration of Real-Time Data:** I successfully integrated real-time stock market data using APIs like Yahoo Finance (`yfinance`) to inform portfolio rebalancing decisions.
*   **LLM Selection and Evaluation:** I compared and evaluated different large language models (LLMs) from OpenAI (GPT-4) and Groq (LLaMA3-8B and LLaMA3-70B) for financial analysis tasks, assessing their strengths and weaknesses in terms of accuracy, speed, and cost.
*   **Agent Framework Development:** I designed and implemented a LangChain agent that dynamically selects tools based on user queries, ensuring efficient workflow and decision-making.
*   **Prompt Engineering:** I refined prompts to guide LLMs in generating actionable and accurate financial recommendations, improving output quality through iterative testing and refinement.
*   **Performance Analysis:** I analyzed the performance of different LLMs in terms of latency and accuracy, providing insights into their suitability for various financial analysis scenarios.
*   **Portfolio Rebalancing Logic:** I implemented a portfolio rebalancing strategy based on equal-weight distribution, ensuring that the AI assistant can provide actionable recommendations for maintaining optimal portfolio balance.

By combining these skills, I created a tool that automates portfolio analysis and rebalancing, offering insights for financial decision-making.

---

## **Implementation Details**

The AI financial portfolio rebalancer was built using LangChain’s agent framework, integrating three core tools for real-time data analysis and decision-making:

### **Tools**

*   **Stock Price Lookup (`get_stock_price`)**  
    *   **Functionality**: Fetches real-time stock prices, daily changes, and percentage changes using Yahoo Finance (`yfinance`).  
    *   **Key Features**:  
        *   Fallback to historical data if metadata is unavailable.  
        *   Error handling for invalid symbols or delisted stocks.  
    *   **Example Output**:  
        ```
        {'symbol': 'AAPL', 'price': 189.84, 'change': 1.23, 'change_pct': 0.65}  
        ```

*   **Portfolio Rebalancer (`rebalance_portfolio`)**  
    *   **Functionality**: Analyzes portfolio weights and suggests buy/sell actions to achieve equal-weight distribution.  
    *   **Key Features**:  
        *   Validates portfolio weight sum (must total ~1.0).  
        *   Generates actionable recommendations (e.g., "Sell AAPL: Decrease weight by 0.1667").  

*   **Market Trend Analysis (`market_trend_analysis`)**  
    *   **Functionality**: Evaluates S&P 500 (via SPY ETF) trends over five days, including returns and volatility.  
    *   **Key Metrics**:  
        *   5-day return  
        *   Annualized volatility  

### **Agent Framework**

*   **LLM Integration**: Tested with OpenAI’s GPT-4, Groq’s LLaMA3-8B, and LLaMA3-70B.  
*   **Workflow**:  
    *   **Market Analysis**: Always invoked first to assess broader market conditions.  
    *   **Individual Stock Price Checks**: Each symbol in the portfolio is queried separately.  
    *   **Rebalancing**: Recommendations generated based on equal-weight strategy.  
*   **Code**:  
    ```
    # Simplified agent initialization  
    agent = initialize_agent(  
        tools=[stock_price_tool, rebalance_tool, trend_tool],  
        llm=llm,  
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
    )  
    ```

---

## **Challenges and Solutions**

*   **Challenge 1: Tool Compatibility Errors**  
    *   **Issue**: `AttributeError: 'Tool' object has no attribute 'is_single_input'` arose due to LangChain’s expectation of Pydantic schemas for multi-input tools.  
    *   **Solution**:  
        *   Defined explicit `args_schema` using Pydantic models.  
        *   Switched to `AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` to support structured inputs.  

*   **Challenge 2: Incorrect Stock Data**  
    *   **Issue**: Frequent 404 errors from Yahoo Finance (e.g., `$'GOOGL': possibly delisted`).  
    *   **Solution**:  
        *   Added error fallbacks using historical data.  
        *   Implemented input validation to reject invalid symbols.  

*   **Challenge 3: Unreliable LLM Outputs**  
    *   **Issue**: Groq LLaMA3-8B produced incomplete or hallucinated recommendations (e.g., "Agent stopped due to iteration limit").  
    *   **Solution**:  
        *   **Prompt Engineering**: Enforced strict workflow order and individual stock checks.  
        *   **Response Formatting**: Required outputs to include numerical adjustments (e.g., "Decrease weight by 0.1667").  

*   **Challenge 4: Performance Variability**  
    *   **Issue**: GPT-4 produced accurate but slow responses (~22 seconds), while Groq LLaMA3-70B was faster (~6 seconds) but occasionally less precise.  
    *   **Solution**:  
        *   Cached market trend data to reduce API calls.  
        *   Limited agent iterations to prevent timeouts.  

---

## **Analysis of LLM Strengths and Weaknesses for Financial Tasks**

Based on the test results, each LLM demonstrates unique strengths and weaknesses for financial portfolio analysis tasks.

*   **OpenAI GPT-4**  
    *   **Strengths**: OpenAI's GPT-4 is basic in tool selection and quality of advice. It mostly accurately calculated portfolio rebalancing actions and identified balanced portfolios.  
    *   **Weaknesses**: In some edge cases (e.g., a portfolio consisting solely of Bitcoin), GPT-4 struggled to understand the broader context, leading to less useful recommendations. The cost per query is also higher.

*   **Groq LLaMA3-8B**  
    *   **Strengths**: The Groq LLaMA3-8B model offers low latency, making it suitable for rapid analysis and real-time applications. It was the only model that did best in context understanding (albeit in only one case) where it correctly used the other tools in combination with the primary rebalancing tool. It could correctly identify even the edge case.  
    *   **Weaknesses**: The model stopped due to iteration or time limits. Seems it lacks the reliability to be deployed for critical financial decisions.

*   **Groq LLaMA3-70B**  
    *   **Strengths**: This model strikes a balance between speed and accuracy, delivering responses faster than GPT-4 while maintaining a better level of correctness. It even identified the edge case correctly. The action plan is also relatively easy to interpret  
    *   **Weaknesses**: In the almost balanced case, it failed to use other tools before making a recommendation.

LLaMA3-8B model is the best in understanding larger context and task and purpose, and using the right tools, though it failed due to iteration limits. If this issue could be resolved, I would choose LLaMA3-8B model. LLaMA3-70B offers a solid tradeoff of speed and cost for real-time analysis and could be ranked second.

---

## **Recommendations for Financial Analysis Scenarios**

Based on the analysis of OpenAI GPT-4, Groq LLaMA3-8B, and Groq LLaMA3-70B, it is evident that each LLM has distinct strengths and weaknesses, making them suitable for different financial analysis scenarios. While OpenAI GPT-4 demonstrates high accuracy and reasoning capabilities, Groq's models offer faster response times and cost-effective solutions. Below are recommendations tailored to specific use cases:

### **Scenario-Based Recommendations**

| **Scenario**                      | **Recommended LLM** | **Rationale**                                              |
|-----------------------------|--------------------|------------------------------------------------------------|
| High-Stakes Rebalancing     | Groq LLaMA3-8B      | Accuracy and nuanced reasoning critical for large portfolios. |
| Real-Time Analysis          | Groq LLaMA3-70B     | Speed and reliability for time-sensitive decisions.         |
| Simple Portfolios           | Groq LLaMA3-8B      | Cost-effective for basic checks (e.g., 2–3 assets).         |
| Educational Use             | Groq LLaMA3-70B     | Balances speed and clarity for student use cases.           |

---

## **Setup Instructions**

### **Prerequisites**
Before you begin, ensure you have the following installed:
- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **Pip**: Comes pre-installed with Python.
- **Virtual Environment (Recommended)**: To isolate dependencies.

---

### **Installation**

1. **Create a Virtual Environment** (Optional but recommended):
python -m venv langchain_env
source langchain_env/bin/activate # On Windows: langchain_env\Scripts\activate

text

2. **Install Required Packages**:
Run the following command to install all necessary dependencies:
pip install langchain langchain-openai langchain-groq langchain-community requests pandas yfinance

text

3. **Verify Installation**:
Test the installation by importing the required libraries in a Python script:
import langchain
import yfinance

text

---

## **Step-by-Step Guide for API Setup and Model Initialization**

### **Setting Up OpenAI API**

1. **Create an Account**:
- Visit [OpenAI Platform](https://platform.openai.com/) and sign up for an account if you don’t already have one.

2. **Get Your API Key**:
- Go to your account settings.
- Navigate to "API Keys" ([OpenAI API Keys](https://platform.openai.com/account/api-keys)).
- Click "Create new secret key" and copy the generated key (it will only be shown once).

3. **Store the API Key Securely**:
- Create a `.env` file in your project directory.
- Add the following line to the file:
  ```
  OPENAI_API_KEY=your_key_here
  ```
- Add `.env` to your `.gitignore` file to prevent accidental sharing.

4. **Initialize OpenAI in Code**:
from langchain_openai import ChatOpenAI

openai_llm = ChatOpenAI(
model_name="gpt-4", # or "gpt-3.5-turbo" for a less expensive option
temperature=0,
api_key=os.getenv("OPENAI_API_KEY")
)

text

---

### **Setting Up Groq API**

1. **Create an Account**:
- Visit [Groq Console](https://groq.com/) and sign up for an account.

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

text

---

### **Setting Up Yahoo Finance API**

No API key is required for Yahoo Finance integration! The `yfinance` library is used to fetch stock data directly.

Example usage:
import yfinance as yf

stock = yf.Ticker("AAPL")
data = stock.history(period="1d")
print(data)

text

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

## **Running the Application**

1. Ensure your `.env` file contains valid API keys for OpenAI and Groq.
2. Activate your virtual environment (if used):
    ```
    source langchain_env/bin/activate  # On Windows: langchain_env\Scripts\activate
    ```
3. Run your Python script or Jupyter Notebook.

---

## **Troubleshooting Tips**

- Ensure all dependencies are installed correctly by running `pip list`.
- Verify that your `.env` file contains valid API keys.
- If encountering issues with Yahoo Finance data, ensure that the stock ticker symbols are valid and not delisted.
- For OpenAI-related issues, confirm that your API key has sufficient usage limits.

---

## **References**

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Platform](https://platform.openai.com/)
- [Groq Console](https://groq.com/)
- [Yahoo Finance Python Library (`yfinance`)](https://pypi.org/project/yfinance/)

---

<div align="center">
  <h4>Happy Coding! 🚀</h4>
</div>
