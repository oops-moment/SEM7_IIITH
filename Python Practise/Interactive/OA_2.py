class Solution:
    def solve(self,rows : list[str]):
        valid_currency={"EUR","USD","SGD","BRL","JPY","ISK","KRW"}
        current_file=None
        current_headers=None
        final_dict={}
        withdrawn={}

        for row in rows:
            if(row.endswith(".csv")):
                current_file,date=row.split('_')
                current_headers=None
            else:
                if(',' in row):
                    if current_headers==None:
                        current_headers=row.split(',')
                    else :
                        values=row.split(',')
                        data=dict(zip(current_headers,values))

                        if(data['currency'] not in valid_currency or 
                           not data['amount'].isdigit() or not data['evidence_due_by'].isdigit()):
                            continue
                        
                        dispute_ID=f"{current_file}{data['transaction']}"

                        if(data['reason']=="withdrawn"):
                            if dispute_ID not in withdrawn or date > withdrawn[dispute_ID]:
                                withdrawn[dispute_ID]=date

                        else:
                            if dispute_ID not in final_dict or date < final_dict[dispute_ID]['date']:
                                final_dict[dispute_ID]={
                                    'merchant':data['merchant'],
                                    'amount':data['amount'],
                                    'currency':data['currency'],
                                    'evidence_due_by':data['evidence_due_by'],
                                    'date':date
                                }
    
        for disputeID,date in withdrawn.items():
            if dispute_ID in final_dict and date > final_dict[dispute_ID]['date']:
                del final_dict[dispute_ID]
        
        out = []
        for te, data in sorted(final_dict.items()):
            amount = int(data['amount'])
            if data['currency'] in ['EUR', 'USD', 'SGD', 'BRL']:
                amount = f"{amount/100:.2f}"
            else:
                amount = f"{amount}.00"
            out.append(f"{te},{data['merchant']},{amount}{data['currency']},{data['evidence_due_by']}")

        return '\n'.join(out)




if __name__ == '__main__':
    solution=Solution()
    # n=int(input("Enter number of Entries "))
    # rows=[]
    # for _ in range(n):
    #     row=input("Enter the chargeback ")
    #     rows.append(row)
    rows = [
        "VISA_20230601.csv",
        "transaction,merchant,amount,currency,evidence_due_by,reason",
        "123890132,47821,37906,USD,1686812400,fraudulent",
        "110450953,63724,12750,JPY,1686898800,duplicate",
        "JCB_20230604.csv",
        "transaction,merchant,currency,amount,evidence_due_by,reason",
        "110450953,11000,SGD,15000,1686898820,duplicate"
    ]
    result=solution.solve(rows)

    print(result)