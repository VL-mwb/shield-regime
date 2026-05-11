"""
Model Context Protocol (MCP) Server for ShieldRegime.
This allows any MCP-compatible AI agent (Claude Desktop, etc.) to natively use ShieldRegime.
"""
from mcp.server.fastmcp import FastMCP
import pandas as pd
import yfinance as yf
from shield_regime.detectors.pump_dump import PumpDumpDetector
from shield_regime.detectors.wash_trade import WashTradeDetector

# Initialize the FastMCP server
mcp = FastMCP("ShieldRegime Surveillance Server")

@mcp.tool()
def scan_ticker_for_manipulation(ticker: str, start_date: str, end_date: str) -> str:
    """
    Run a full RegTech market surveillance scan on a given financial ticker to detect 
    Pump-and-Dump schemes and Wash Trading anomalies.
    
    Args:
        ticker: The financial ticker symbol (e.g., 'DOGE-USD', 'GME', 'AAPL').
        start_date: The start date in YYYY-MM-DD format.
        end_date: The end date in YYYY-MM-DD format.
        
    Returns:
        A structured string report detailing the detected anomalies and alert severities.
    """
    try:
        # 1. Fetch data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return f"Error: No data found for ticker {ticker} in the specified date range."
            
        # Convert column names to lowercase to be compatible with ShieldRegime detectors
        df_clean = pd.DataFrame(index=df.index)
        if isinstance(df.columns, pd.MultiIndex):
            close_col = ("Close", ticker)
            vol_col = ("Volume", ticker)
            high_col = ("High", ticker)
            low_col = ("Low", ticker)
            df_clean["close"] = df[close_col].astype(float)
            df_clean["volume"] = df[vol_col].astype(float)
            df_clean["high"] = df[high_col].astype(float)
            df_clean["low"] = df[low_col].astype(float)
        else:
            df_clean["close"] = df["Close"].astype(float)
            df_clean["volume"] = df["Volume"].astype(float)
            df_clean["high"] = df["High"].astype(float)
            df_clean["low"] = df["Low"].astype(float)
            
        report = [f"🛡️ ShieldRegime Market Surveillance Audit for {ticker}"]
        report.append(f"Period: {start_date} to {end_date}")
        report.append("-" * 50)
        
        # 2. Run Pump-and-Dump Detection
        pd_detector = PumpDumpDetector()
        pd_alerts = pd_detector.scan(df_clean, ticker=ticker)
        
        report.append(f"[Pump & Dump Scan]: Found {len(pd_alerts)} alerts.")
        for alert in pd_alerts:
            report.append(f"  - [{alert.severity.name}] Date: {alert.date.strftime('%Y-%m-%d')} | Info: {alert.message}")
            
        # 3. Run Wash Trade Detection
        wt_detector = WashTradeDetector()
        wt_alerts = wt_detector.scan(df_clean, ticker=ticker)
        
        report.append(f"\n[Wash Trading Scan]: Found {len(wt_alerts)} alerts.")
        for alert in wt_alerts:
            report.append(f"  - [{alert.severity.name}] Date: {alert.date.strftime('%Y-%m-%d')} | Info: {alert.message}")
            
        if len(pd_alerts) == 0 and len(wt_alerts) == 0:
            report.append("\n✅ CLEAR: No market manipulation anomalies detected.")
        else:
            report.append("\n⚠️ WARNING: Suspected market manipulation identified. Recommend suspending automated capital deployment.")
            
        return "\n".join(report)

    except Exception as e:
        return f"Audit failed due to an internal error: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server using standard stdio transport (required for AI agent integrations)
    mcp.run(transport='stdio')
