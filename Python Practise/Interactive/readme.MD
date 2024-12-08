Global Startup Name Validator

Problem Statement

You are tasked with building a Global Startup Name Validator for a platform that helps entrepreneurs register new company names across multiple countries. Similar to existing platforms, the system needs to ensure that no two companies are registered with names that are too similar, to avoid confusion or legal issues. However, different countries have specific rules for validating company names.

Your Task
Implement the function validate_name_availability, which takes in a list of account IDs and corresponding company name requests.
For each requested company name, determine whether it is Name Available or Name Not Available, based on the following conditions:
The first instance of a valid name should be marked as available, and subsequent instances of similar names should be marked as unavailable.
Once a name is marked as unavailable, it cannot be registered by another company.
Name Comparison Rules
Names do not have to be identical to be considered the same. Apply the following rules to determine if two company names are similar:

Case Insensitivity: 
- Ignore differences in case, e.g., "Innovate Tech" is the same as "INNOVATE TECH".
- Special Characters: Treat characters like commas (,), periods (.), ampersands (&), and hyphens (-) as spaces. For example, "Innovate-Tech LLC" is the same as "Innovate Tech Inc."
- Multiple Spaces: Treat multiple consecutive spaces as a single space.
- Ignored Words: Ignore leading articles such as "The", "A", and "An" (e.g., "The Global Enterprise" is the same as "Global Enterprise").
- Country-Specific Suffixes: Suffixes that denote legal structures (like "Inc.", "Ltd.", "GmbH", "S.A.", etc.) should be ignored in the comparison. Assume suffixes can be from a list: ["Inc.", "Corp.", "Ltd.", "LLC", "GmbH", "S.A.", "Pty Ltd", "L.L.C."].
- Numerical Names: Names with numbers at the end, like "Tech Solutions 123" and "Tech Solutions", should be considered distinct unless the numbers refer to the year of foundation (e.g., "Tech Solutions 2020" and "Tech Solutions 2021" are distinct).
- Abbreviations: Common abbreviations should be expanded and compared. For example, "Co." should be treated as "Company", and "Intl." as "International".
Name Collapse: After applying the transformations, if the resulting name is empty or contains only spaces, it should be considered not available.
Input
Each input will represent a name availability check with an account ID (e.g., acct_abc123) and a proposed company name, separated by a | (pipe) character.

Output
For each company name request, print a line with the account ID, followed by a pipe (|), followed by either Name Available or Name Not Available, based on the availability of the name.

Additional Considerations
Names from different countries may still conflict if they follow the same rules. The validator should handle these conflicts globally, across all countries.
A name that is marked as unavailable should not be considered available for any future request.
Example 1 - Single Entry
Input:

Copy code
acct_00123|Innovate Tech Inc.
Output:

Copy code
acct_00123|Name Available
Explanation: The name "Innovate Tech Inc." is available and marked as such.

Example 2 - Matching Entries
Input:

Copy code
acct_00123|Innovate Tech Inc.
acct_00567|INNOVATE-TECH LTD.
Output:

mathematica
Copy code
acct_00123|Name Available
acct_00567|Name Not Available
Explanation: The names "Innovate Tech Inc." and "INNOVATE-TECH LTD." are considered the same after applying the transformation rules. The second request is unavailable.

Example 3 - Multiple Suffixes and Articles
Input:

sql
Copy code
acct_00123|The Global Company Inc.
acct_00567|Global Company Ltd.
acct_00987|Global Company International LLC.
Output:

mathematica
Copy code
acct_00123|Name Available
acct_00567|Name Not Available
acct_00987|Name Not Available
Explanation: All names are considered too similar after ignoring leading articles and suffixes, so only the first name is marked available.

Example 4 - Name Collapse
Input:

Copy code
acct_00123|The, Inc.
acct_00567|The Tech, LLC.
Output:

mathematica
Copy code
acct_00123|Name Not Available
acct_00567|Name Available
Explanation: "The, Inc." collapses to an empty name after ignoring the article "The" and suffix "Inc.", making it unavailable. "The Tech, LLC." becomes "Tech", which is valid and available.

Example 5 - Numerical Distinction
Input:

yaml
Copy code
acct_00123|Tech Solutions 2020 Inc.
acct_00567|Tech Solutions 2021 LLC.
Output:

Copy code
acct_00123|Name Available
acct_00567|Name Available
Explanation: The names "Tech Solutions 2020" and "Tech Solutions 2021" are considered distinct due to the different numerical endings, so both are available.

Good luck!

arduino
Copy code

This new variant includes additional checks for country-specific suffixes, number handling, and abbreviation expansions. It will provide a more comprehensive challenge!




--------------------------------------------------------------------

Here’s a modified version of your original question with added complexity:

Account Balance Manager: StripePay Backend (Advanced)

Imagine you are building the backend for an app called StripePay, a popular peer-to-peer payment system. StripePay users send money, manage multiple bank accounts, and schedule future transactions. Our system handles a high volume of transactions, but we also need to ensure proper execution order, even with network delays and scheduled transactions.

Your task is to implement core backend functionality for StripePay, including handling transactions between users, bank account management, and reading the account balance.

Requirements

You need to support the following commands:

	•	INIT
	•	Usage: INIT,name,balance,bank_1,bank_2,...,bank_n
	•	Initialize a user’s account with the given balance and a list of banks they are connected to.
	•	Always comes before any other command for that user.
	•	No response value.
	•	POST
	•	Usage: POST,timestamp,sender,receiver,amount
	•	sender sends amount to receiver at timestamp.
	•	If sender or receiver is a bank, handle it as a deposit/withdrawal.
	•	Return FAILURE if the transaction fails:
	•	The sender’s balance would go negative.
	•	A bank is not connected to the sender or receiver (e.g., withdrawal from an unlinked bank).
	•	Both sender and receiver are banks.
	•	Either user or bank doesn’t exist (wasn’t initialized).
	•	Otherwise, return SUCCESS.
	•	GET
	•	Usage: GET,timestamp,name
	•	Return the account balance for name at timestamp.
	•	If name doesn’t exist, return FAILURE.
	•	SCHEDULE
	•	Usage: SCHEDULE,timestamp,execute_time,sender,receiver,amount
	•	Schedule a transaction to be executed at execute_time.
	•	The same rules apply as in POST when the transaction is actually executed.
	•	If scheduling is successful, return SCHEDULED.
	•	If it fails (e.g., invalid user, invalid bank), return FAILURE.
	•	ADD_BANK
	•	Usage: ADD_BANK,timestamp,name,bank
	•	Link a new bank account to a user at timestamp.
	•	Return SUCCESS if the bank is successfully added.
	•	If the bank is already linked or if the user doesn’t exist, return FAILURE.
	•	REMOVE_BANK
	•	Usage: REMOVE_BANK,timestamp,name,bank
	•	Remove a linked bank account from a user.
	•	Return SUCCESS if the bank is successfully removed.
	•	If the bank is not linked or if the user doesn’t exist, return FAILURE.

Additional Constraints:

	1.	Concurrency and Chronology:
	•	Transactions must be executed in the order of timestamp, regardless of the order they arrive in.
	•	Scheduled transactions should only be executed after their execute_time passes.
	2.	Edge Cases:
	•	Transactions with identical timestamps should be processed in the order they appear.
	•	All commands should process efficiently, even with thousands of transactions per second.

-----

Coding Round The interviewer was fair enough. I just didn't complete enough parts of the question. The frustrating part was 126 lines of markdown explaining the question. Unreasonable to expect someone to parse through that long of a
problem description for a 30min problem. Given a String, split it into major parts separated by special char /'. For each major part that's split by '/', we can
further split it into minor parts separated by'!.
#### Example 1
# str = stripe.com/payments/checkout/customer.john.doe
# minor_parts = 2
# after Part 1 compression
#=>
# s4e.c1m/pos/c6t/cr.j2n.dle
# after Part 2 compression
#=>
# s4e.c1m/p6s/c6t/c6r.j5e
#### Example 2
# Given:
# str = www.api.stripe.com/checkout
# minor_parts = 3
# (after Part 1 compression)
# =>
# w1w.a1i.s4e.c1m/c6t
# (then after Part 2 compression)
#=>
# w1w.ali.s/m/c6t

The question is to compress the string. The rule is that, for each major part, it contains at most <minor_parts> minor parts. And for each minor parts, you need to compress it in the format of <heading_letter><letter_count_in_betwee><tailing_letter>. The edge case is, what if the minor part contains less than 3 letters? Like the first minor part of m.stripe.com/checkout.

_______

art 1: Direct Shipping Cost
Given an input string that describes various shipping routes and their associated costs, we need to extract the direct cost from a given origin (X) to a destination (Y). The input string contains shipping routes in the format:

"US,UK,UPS,5:US,CA,FedEx,3:CA,UK,DHL,7"
Here, each route is separated by a colon (:), and each route consists of:

Origin (e.g., US)
Destination (e.g., UK)
Shipping Company (e.g., UPS)
Shipping Cost (e.g., 5)
We want to find the cost to ship an item directly from X to Y.

Part 2: Indirect Shipping Cost
In this part, we need to calculate the cost if the shipment takes an intermediate route. For example, shipping from X to Y via another location (Z). For instance, shipping from US to UK via CA would have the cost: US -> CA cost + CA -> UK cost.

_____

https://leetcode.com/discuss/interview-question/4595354/Stripe-Intern-Interview-Question

Part 1: Parse a string in the format "USD:CAD:DHL:5,USD:GBP:FEDX:10", representing currency conversion rates from a source to a target currency and the associated shipping method. Write a method to convert a given amount from one currency to another. Only direct conversions are allowed.

Part 2: Write a method that returns the cost and shipping methods involved, allowing at most one hop in the conversion from one currency to another.

Part 3: Write a method that returns the minimum cost and involved shipping methods, allowing at most one hop for the conversion.

You are given a Python class Inventory, which is supposed to simulate an inventory management system for a store. The class should allow users to:
	1.	Add items to the inventory with a specified price and quantity.
	2.	Remove items from the inventory, reducing the quantity or printing an error if there’s not enough stock.
	3.	Calculate the total value of all items in the inventory.
	4.	Check the quantity of a specific item in the inventory.

However, the code has a few bugs and inefficiencies. Your task is to: