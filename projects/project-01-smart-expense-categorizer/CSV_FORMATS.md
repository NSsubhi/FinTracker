# 📋 Supported CSV Formats

The Smart Expense Categorizer **automatically detects** various CSV formats! You don't need to stick to a specific format.

## ✅ Automatic Detection Features

- **Auto-detects column names** (many variations supported)
- **Handles different date formats**
- **Removes currency symbols** (₹, $, €, etc.)
- **Handles commas in numbers** (1,000.50 → 1000.50)
- **Detects debit/credit** from column names or amount signs
- **Works with multiple encodings** (UTF-8, Latin-1, etc.)

---

## 📊 Supported Column Name Variations

### Date Column (Any of these will work):
- `Date`, `Transaction Date`, `Trans Date`
- `Date/Time`, `Date_Time`, `Posting Date`
- `Value Date`, `Txn Date`, `TransactionDate`
- Or any column containing "date" or "time"

### Description Column (Any of these will work):
- `Description`, `Transaction Description`, `Details`
- `Narration`, `Memo`, `Remarks`, `Notes`
- `Particulars`, `Payment Details`
- Or any text column (auto-detected)

### Amount Column (Any of these will work):
- `Amount`, `Transaction Amount`, `Amt`
- `Value`, `Sum`, `Transaction Value`
- Or any numeric column

### Debit/Credit Columns:
- Separate columns: `Debit` / `Credit`
- Or: `Withdrawal` / `Deposit`
- Or: `Paid` / `Received`
- Or: `Expense` / `Income`

### Transaction Type:
- `Type`, `Transaction Type`, `Category`
- Values: DEBIT, CREDIT, D, C, Withdrawal, Deposit, etc.

---

## 📝 Example CSV Formats That Work

### Format 1: Standard Format
```csv
Date,Description,Amount,Transaction_Type
2024-01-01,Payment to Zomato,500,DEBIT
2024-01-02,Salary Credit,50000,CREDIT
```

### Format 2: Bank Statement Format
```csv
Transaction Date,Narration,Debit,Credit
01/01/2024,Payment to Zomato,500,
02/01/2024,Salary Deposit,,50000
```

### Format 3: Alternative Column Names
```csv
Posting Date,Details,Transaction Amount,Type
2024-01-01,Amazon Purchase,2500,DEBIT
2024-01-02,Investment Income,3000,CREDIT
```

### Format 4: Separate Debit/Credit Columns
```csv
Date,Description,Withdrawal,Deposit
2024-01-01,Grocery Shopping,1200,
2024-01-02,Salary Credit,,50000
```

### Format 5: Just Date, Description, Amount
```csv
Date,Description,Amount
2024-01-01,Restaurant Bill,850
2024-01-02,Electricity Bill,-1100
```
*(Negative amounts = DEBIT, Positive = CREDIT)*

### Format 6: Currency Symbols Included
```csv
Date,Description,Amount
2024-01-01,Payment to Zomato,₹500
2024-01-02,Salary Credit,$5,000.00
```
*(Currency symbols automatically removed)*

---

## 🎯 What the System Does Automatically

1. **Column Detection**: Finds date, description, and amount columns automatically
2. **Date Parsing**: Handles various date formats (DD/MM/YYYY, MM-DD-YYYY, etc.)
3. **Amount Cleaning**: 
   - Removes currency symbols (₹, $, €, £)
   - Removes commas (1,000 → 1000)
   - Handles decimals (.50 or 0.50)
4. **Transaction Type Detection**:
   - From separate Debit/Credit columns
   - From Type column
   - From amount sign (negative = debit, positive = credit)
5. **Encoding**: Tries UTF-8, Latin-1, and CP1252 automatically

---

## ❌ What Won't Work

- **Missing Required Columns**: Must have Date, Description, and Amount (or Debit/Credit)
- **Completely Unstructured Data**: Needs at least some column structure
- **Binary Formats**: Excel files (.xlsx) - need to export as CSV first

---

## 💡 Tips

1. **Export from Bank**: Most bank exports work automatically!
2. **Excel Files**: Export as CSV first (File → Save As → CSV)
3. **Multiple Currencies**: Currency symbols are removed automatically
4. **Mixed Formats**: The system handles mixed date formats in the same file

---

## 🧪 Test Your CSV

1. Upload your CSV file
2. If auto-detection fails, you'll see a helpful error message
3. The system will tell you what columns it detected

---

**The system is smart - just upload your CSV and it will figure out the format! 🚀**

