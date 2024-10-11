from datetime import datetime

def parseNetworkChargebackInfo(rows):
    d = {}
    we = {}
    vC = {"EUR", "USD", "SGD", "BRL", "JPY", "ISK", "KRW"}
    
    cF = None
    he = None
    
    for row in rows:
        if row.endswith('.csv'):
            cF = row
            network, date_str = cF.split('_')
            date = datetime.strptime(date_str.split('.')[0], '%Y%m%d')
            he = None
        elif ',' in row:
            if he is None:
                he = row.split(',')
            else:
                values = row.split(',')
                data = dict(zip(he, values))
                
                if (data['currency'] not in vC or
                    not data['amount'].isdigit() or
                    not data['evidence_due_by'].isdigit()):
                    continue
                
                te = f"{network}{data['transaction']}"
                
                if data['reason'] == 'withdrawn':
                    if te not in we or date > we[te]:
                        we[te] = date
                else:
                    if te not in d or date < d[te]['date']:
                        d[te] = {
                            'merchant': data['merchant'],
                            'amount': int(data['amount']),
                            'currency': data['currency'],
                            'evidence_due_by': data['evidence_due_by'],
                            'date': date
                        }
    for te, wD in we.items():
        if te in d and wD > d[te]['date']:
            del d[te]
    out = []
    for te, data in sorted(d.items()):
        amount = data['amount']
        if data['currency'] in ['EUR', 'USD', 'SGD', 'BRL']:
            amount = f"{amount/100:.2f}"
        else:
            amount = f"{amount}.00"
        out.append(f"{te},{data['merchant']},{amount}{data['currency']},{data['evidence_due_by']}")
    return '\n'.join(out)


if __name__ == "__main__":
    rows = [
        "VISA_20230601.csv",
        "transaction,merchant,amount,currency,evidence_due_by,reason",
        "123890132,47821,37906,USD,1686812400,fraudulent",
        "110450953,63724,12750,JPY,1686898800,duplicate",
        "JCB_20230604.csv",
        "transaction,merchant,currency,amount,evidence_due_by,reason",
        "110450953,11000,SGD,15000,1686898820,duplicate"
    ]
    result = parseNetworkChargebackInfo(rows)
    print(result)
