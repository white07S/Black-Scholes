# Black-Scholes

Implementation of Black Scholes model in python and tested on Options

**Its important to understand that options trading and its algorithms depend on caching and spread so this profit and performance can be increased if this algorithm is written in C or C++**

Black-Scholes is a pricing model used to determine the fair price or **theoretical value for a call or a put option based on six variables** such as volatility, type of option, underlying stock price, time, strike price, and risk-free rate.

![image](https://user-images.githubusercontent.com/58583011/176910703-66a35f39-e00d-4b4b-9153-400904b15d4b.png)

* C	=	call option price
* N	=	CDF of the normal distribution
* S_t	=	spot price of an asset
* K	=	strike price
* r	=	risk-free interest rate
* t	=	time to maturity
* \sigma	=	volatility of the asset

The formula can be interpreted by first decomposing a call option into the difference of two binary options: 
an asset-or-nothing call minus a cash-or-nothing call (long an asset-or-nothing call, short a cash-or-nothing call). 
A call option exchanges cash for an asset at expiry, while an asset-or-nothing call just yields the asset (with no cash in exchange) and a cash-or-nothing call just yields cash (with no asset in exchange). 
The Black–Scholes formula is a difference of two terms, and these two terms equal the values of the binary call options. These binary options are much less frequently traded than vanilla call options, but are easier to analyze.

**Dataset**
![Screenshot from 2022-07-02 12-37-47](https://user-images.githubusercontent.com/58583011/176997599-dd8d07b0-bc6d-4433-a486-da976414c3da.png)

The model is widely employed as a useful **approximation** to reality, but proper application requires understanding its **limitations**

**For more information**
* https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

