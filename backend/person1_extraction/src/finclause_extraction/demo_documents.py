"""Curated demo documents for one-click demos.

All institutions, names, numbers and reference IDs below are **fictional**. The
documents are written to follow the *structure and wording style* of real,
publicly available document types (not copied from any specific institution):

* a bank personal-loan agreement (numbered clauses, schedules of charges);
* an NBFC Key Fact Statement in the tabular format introduced by RBI for retail loans;
* a co-operative bank sanction letter (typically received on paper and scanned);
* health-insurance policy key terms (sum insured, waiting periods, co-payment, exclusions,
  free-look cancellation) as found in retail health policies sold via aggregators;
* e-commerce "No Cost EMI" terms (interest given as an upfront discount, GST on interest,
  bank processing fee, effect of cancellations/returns).

Real institutions' documents are deliberately not reproduced: their exact numbers change
over time and attributing invented figures to them would be misleading.

Each document deliberately exercises pipeline features: running headers and page numbers
(header/footer removal), a word hyphenated across lines (preprocessing), clauses that
continue across pages, nested sub-clauses, tables, an ambiguous date, charges whose amount
is not stated, and discretionary-change clauses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .samples import Line, build_pdf, build_scanned_pdf

B = {"bold": True, "gap": 9}
TITLE = {"bold": True, "size": 13, "gap": 12}


@dataclass(frozen=True)
class DemoDocument:
    doc_id: str
    title: str
    institution: str
    category: str  # loan | insurance | emi
    description: str
    pages: list[list[Line]] = field(hash=False)
    header: Optional[str] = None
    default_scanned: bool = False
    highlights: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return f"{self.doc_id}.pdf"

    def build(self, scanned: Optional[bool] = None, seed: int = 11) -> bytes:
        # (Running headers are only used for multi-page documents: on a single page a header
        # cannot be told apart from body text, exactly as in real documents.)
        """Build the PDF. ``scanned=True`` gives a photocopy-style image-only PDF that requires OCR."""
        scanned = self.default_scanned if scanned is None else scanned
        if scanned:
            return build_scanned_pdf(None, self.pages, dpi=200, style="photocopy", seed=seed,
                                     header=self.header, footer_page_numbers=len(self.pages) > 1)
        return build_pdf(None, self.pages, header=self.header, footer_page_numbers=len(self.pages) > 1)


NORTHWIND_LOAN = DemoDocument(
    doc_id="northwind_bank_personal_loan_agreement",
    title="Personal Loan Agreement",
    institution="Northwind Bank Ltd. (fictional bank)",
    category="loan",
    description="Two-page bank loan agreement: numbered clauses and sub-clauses, tiered foreclosure charges, "
                "penal interest, events of default, cooling-off cancellation.",
    header="Northwind Bank Ltd.  |  Personal Loan Agreement  |  SPECIMEN - fictional document for demonstration",
    highlights=("hyphenation across lines", "cross-page clause", "tiered foreclosure (a)/(b)", "ambiguous date",
                "charge without amount", "default & credit-bureau clauses"),
    pages=[
        [
            ("PERSONAL LOAN AGREEMENT", TITLE),
            "This Loan Agreement is made at Pune on 12/02/2025 between Northwind Bank Ltd., a banking company "
            "(hereinafter referred to as the \"Lender\"), and the person named in Schedule I (hereinafter referred "
            "to as the \"Borrower\").",
            ("1. DEFINITIONS AND LOAN DETAILS", B),
            "1.1 The Lender agrees to grant to the Borrower a term loan of Rs. 6,50,000 (Rupees Six Lakh Fifty "
            "Thousand only) (the \"Loan\").",
            "1.2 The Loan shall be repayable over a tenure of 48 months from the date of disbursement in 48 equated "
            "monthly instalments of Rs. 17,230 each.",
            "1.3 The Borrower shall utilise the Loan solely for personal purposes and not for any speculative purpose.",
            ("2. INTEREST", B),
            "2.1 The Loan shall carry interest at the rate of 11.75% per annum (fixed), computed on a monthly "
            "reducing balance basis.",
            "2.2 In the event of any delay in payment of an EMI, the Borrower shall be liable to pay penal interest at "
            "2% per month on the overdue amount from the due date till the date of actual pay-\nment.",
            ("3. FEES AND CHARGES", B),
            "3.1 A processing fee of 1.5% of the Loan amount plus applicable GST, subject to a minimum of Rs. 3,000, "
            "shall be deducted from the disbursement amount and is non-refundable.",
            "3.2 Cheque/NACH bounce charges of Rs. 590 per instance shall be levied for every dishonoured payment "
            "instruction.",
            "3.3 Foreclosure of the Loan is permitted only after payment of 12 EMIs. Foreclosure charges shall be "
            "levied as under:",
            "(a) 4% of the principal outstanding if the Loan is foreclosed between the 13th and 24th EMI;",
            "(b) 2% of the principal outstanding if the Loan is foreclosed after the 24th EMI.",
            "3.4 Part-prepayment charges shall be as per the Schedule of Charges of the Lender prevailing at the time",
        ],
        [
            "of prepayment.",
            "3.5 Stamp duty and documentation charges of Rs. 1,500 shall be borne by the Borrower.",
            ("4. REPAYMENT", B),
            "4.1 The EMI shall be payable on the 7th day of every month by way of NACH mandate. The first EMI shall "
            "fall due on 07/04/2025.",
            "4.2 Any payment received shall be appropriated first towards charges and penal interest, then towards "
            "interest and thereafter towards principal.",
            ("5. EVENTS OF DEFAULT", B),
            "5.1 If the Borrower fails to pay any EMI within a grace period of 5 days from its due date, it shall "
            "constitute an event of default.",
            "5.2 Upon occurrence of an event of default, the Lender shall be entitled to recall the entire outstanding "
            "Loan, which shall become immediately due and payable, and to initiate legal proceedings for recovery.",
            "5.3 The Lender shall be entitled to report the default to credit information companies, which may "
            "adversely affect the credit score of the Borrower.",
            "5.4 The Lender shall have a right of set-off against any deposits of the Borrower held with the Lender.",
            ("6. GENERAL", B),
            "6.1 The Lender reserves the right to revise the fees and charges at its sole discretion with prior "
            "notice of 30 days to the Borrower.",
            "6.2 Notwithstanding anything contained herein, the Borrower may cancel the Loan within a cooling-off "
            "period of 3 days from disbursement by repaying the principal and proportionate interest without any "
            "foreclosure charges.",
        ],
    ],
)

SAHYADRI_KFS = DemoDocument(
    doc_id="sahyadri_finance_key_fact_statement",
    title="Key Fact Statement - Personal Loan",
    institution="Sahyadri Finance Ltd. (fictional NBFC)",
    category="loan",
    description="RBI-style Key Fact Statement with tables: APR, floating rate linked to repo rate, bounce and "
                "penal charges, look-up period exit, insurance premium, recovery agents.",
    highlights=("table layout", "floating rate + repo rate", "APR", "look-up period exit", "recovery & CIBIL"),
    pages=[
        [
            ("KEY FACT STATEMENT (KFS) - PERSONAL LOAN", TITLE),
            ("Sahyadri Finance Ltd. (NBFC)  |  SPECIMEN - fictional document for demonstration", {"gap": 4}),
            "Application No. SF-PL-04417 | Date of KFS: 03-03-2025",
            ("Section A - Details of the Loan", B),
            ("", {"table": [
                ("Loan amount sanctioned", "Rs. 3,00,000"),
                ("Loan tenure", "36 months"),
                ("EMI amount", "Rs. 10,442 per month, payable in 36 EMIs"),
                ("Rate of interest", "15.5% p.a. (floating), linked to the RBI repo rate"),
                ("Annual Percentage Rate (APR)", "17.9% (includes processing fee and insurance)"),
                ("Processing fee", "2.5% of the loan amount + GST"),
                ("Insurance premium (optional)", "Rs. 4,200 single premium for credit life cover"),
                ("Due date of EMI", "5th of every month by NACH auto-debit"),
            ]}),
            ("Section B - Fees and Charges", B),
            ("", {"table": [
                ("Penal charges", "2% per month on the overdue EMI amount"),
                ("Cheque / NACH bounce charges", "Rs. 500 per bounce + GST"),
                ("Foreclosure charges", "5% of principal outstanding"),
                ("Part-prepayment charges", "As per the Schedule of Charges"),
                ("Duplicate NOC/statement charges", "Rs. 250 per request"),
            ]}),
            ("Section C - Other Important Terms", B),
            "1. The rate of interest is linked to the RBI repo rate and shall be reset quarterly. Any change in the "
            "repo rate shall be passed on to the Borrower, and the EMI or tenure may change accordingly.",
            "2. The Lender reserves the right to change the spread over the benchmark at its sole discretion, with "
            "intimation to the Borrower.",
            "3. In case of default, the Lender may engage recovery agents and shall report the default to credit "
            "bureaus including CIBIL.",
            "4. The Borrower may exit the loan within the look-up period of 3 days by paying the principal and "
            "proportionate APR without any penalty.",
            "5. The insurance premium, if opted, is collected on behalf of the insurer and is not refunded after the "
            "free look period of 30 days.",
        ],
    ],
)

KAVERI_SANCTION = DemoDocument(
    doc_id="kaveri_cooperative_bank_sanction_letter",
    title="Sanction Letter - Personal Loan (scanned)",
    institution="Kaveri Co-operative Bank Ltd. (fictional bank)",
    category="loan",
    description="A one-page sanction letter as typically received on paper. The default demo is a skewed, noisy "
                "photocopy scan, so the text must be read by OCR.",
    default_scanned=True,
    highlights=("OCR on a skewed photocopy", "OCR confidence per word", "guarantor requirement", "recall clause"),
    pages=[
        [
            ("KAVERI CO-OPERATIVE BANK LTD.", {"bold": True, "size": 14, "gap": 4}),
            ("Mysuru Main Branch  |  SPECIMEN - fictional document for demonstration", {"gap": 14}),
            ("SANCTION LETTER - PERSONAL LOAN", {"bold": True, "size": 12, "gap": 10}),
            "Ref: KCB/PL/118/25          Date: 18/01/2025",
            "Dear Customer,",
            "We are pleased to inform you that your application for a personal loan has been sanctioned on the "
            "following terms and conditions:",
            "1. Sanctioned amount: Rs. 2,00,000 (Rupees Two Lakh only).",
            "2. Rate of interest: 13.25% per annum (fixed) on reducing balance.",
            "3. Repayment: 24 equated monthly instalments of Rs. 9,525 each, payable on or before the 10th of "
            "every month.",
            "4. Processing charges: 1% of the sanctioned amount plus GST, payable upfront.",
            "5. Delayed payment charges: 24% per annum on the overdue instalment for the period of delay.",
            "6. Prepayment: Allowed after 6 months without any prepayment charges.",
            "7. Security: Personal guarantee of one guarantor acceptable to the Bank.",
            "8. The Bank may recall the loan if any information furnished by you is found to be incorrect.",
            "9. This sanction is valid for 30 days from the date of this letter.",
            "Yours faithfully,",
            "Branch Manager",
        ],
    ],
)

SURAKSHA_HEALTH = DemoDocument(
    doc_id="suraksha_health_insurance_policy_terms",
    title="Health Insurance - Policy Schedule and Key Terms",
    institution="Suraksha General Insurance Co. Ltd. (fictional insurer)",
    category="insurance",
    description="Retail health policy terms: sum insured, premium, room-rent caps, co-payment, waiting periods, "
                "exclusions, grace period, free-look and short-period cancellation.",
    header="Suraksha General Insurance  |  Health Shield  |  SPECIMEN - fictional document for demonstration",
    highlights=("sum insured & premium", "room-rent cap", "co-payment", "waiting periods", "exclusions",
                "free-look cancellation"),
    pages=[
        [
            ("SURAKSHA HEALTH SHIELD - POLICY SCHEDULE AND KEY TERMS", TITLE),
            "Policy No.: SGI-HS-558120 | Policy Period: 01/04/2025 to 31/03/2026 | Plan: Family Floater",
            ("1. COVER", B),
            "1.1 Sum Insured: Rs. 5,00,000 per policy year on a family floater basis.",
            "1.2 Annual Premium: Rs. 18,450 inclusive of GST, payable in advance.",
            "1.3 Room rent is capped at 1% of the Sum Insured per day for a normal room and 2% of the Sum Insured "
            "per day for ICU.",
            "1.4 A co-payment of 20% shall apply on every admissible claim for insured persons aged 61 years and above.",
            "1.5 A no claim bonus of 10% of the Sum Insured shall be added for every claim-free year, up to a "
            "maximum of 50%.",
            ("2. WAITING PERIODS", B),
            "2.1 An initial waiting period of 30 days applies from the policy start date, except for accidents.",
            "2.2 Specified illnesses such as cataract, hernia and joint replacement are covered only after a waiting "
            "period of 24 months.",
            "2.3 Pre-existing diseases shall be covered only after a waiting period of 36 months of continuous coverage.",
            ("3. EXCLUSIONS", B),
            "3.1 The Company shall not be liable to make any payment for expenses on cosmetic or plastic surgery, "
            "unless required due to an accident.",
            "3.2 Treatment arising from alcohol or drug abuse is not covered.",
            "3.3 Expenses on non-medical items such as gloves, toiletries and registration charges are not payable.",
        ],
        [
            ("4. PREMIUM PAYMENT AND RENEWAL", B),
            "4.1 A grace period of 30 days is allowed for payment of the renewal premium. Coverage shall not be "
            "available for any claim during the grace period.",
            ("5. CANCELLATION", B),
            "5.1 You have a free look period of 15 days from receipt of the policy document to cancel the policy if "
            "you disagree with its terms, and the premium shall be refunded after deducting stamp duty and "
            "proportionate risk premium.",
            "5.2 If the Policyholder cancels the policy after the free look period, the refund shall be as per the "
            "short-period scale: 75% of the premium if cancelled within 3 months and 50% if cancelled within 6 months.",
            "5.3 The Company may cancel the policy at any time on grounds of misrepresentation, fraud or "
            "non-disclosure of material facts, without refund of premium.",
        ],
    ],
)

SHOPKART_EMI = DemoDocument(
    doc_id="shopkart_no_cost_emi_terms",
    title="EMI and No Cost EMI Terms",
    institution="ShopKart (fictional e-commerce marketplace)",
    category="emi",
    description="E-commerce EMI terms: No Cost EMI via upfront discount, interest and GST charged by the card "
                "issuer, bank processing fee, effect of cancellation/returns, late payment by the issuer.",
    highlights=("no cost EMI explained", "GST on interest", "bank processing fee", "returns reverse the discount",
                "offer can be withdrawn"),
    pages=[
        [
            ("SHOPKART - EMI AND NO COST EMI TERMS", TITLE),
            ("ShopKart  |  SPECIMEN - fictional document for demonstration", {"gap": 4}),
            "Last updated: 1 January 2025",
            ("1. ABOUT NO COST EMI", B),
            "1.1 No Cost EMI means the interest charged by the issuing bank on your EMI is offered to you as an upfront "
            "discount on the product price at checkout, so that the total EMI payments equal the price of the product.",
            "1.2 The issuing bank will charge interest at 15% per annum on the EMI amount, and this interest is offset "
            "by the upfront discount.",
            "1.3 GST at 18% on the interest component is charged by the issuing bank and is not covered by the discount.",
            "1.4 The issuing bank may charge a processing fee of Rs. 199 plus GST per EMI transaction, which will "
            "appear on your card statement.",
            ("2. ELIGIBILITY AND TENURE", B),
            "2.1 No Cost EMI is available on orders above Rs. 3,000 for EMI tenures of 3, 6 and 9 months on select "
            "credit cards.",
            "2.2 Standard EMI on longer tenures of 12 to 24 months carries interest between 13% and 16% p.a. as per the "
            "card issuer.",
            ("3. CANCELLATIONS AND RETURNS", B),
            "3.1 If the order is cancelled or the product is returned, the upfront discount will be adjusted against "
            "the refund, and the EMI conversion will be cancelled by the bank.",
            "3.2 Any interest already charged by the bank before cancellation will not be refunded by ShopKart.",
            "3.3 Foreclosure of the EMI is subject to the issuing bank's policy and may attract foreclosure charges of "
            "up to 3% of the outstanding principal.",
            ("4. LATE PAYMENT", B),
            "4.1 Missing an EMI payment will attract late payment charges levied by the card issuer, ranging from "
            "Rs. 100 to Rs. 1,300 depending on the outstanding amount, and may affect your credit score.",
            "4.2 ShopKart reserves the right to modify or withdraw the No Cost EMI offer at any time without prior "
            "notice.",
        ],
    ],
)

DEMO_DOCUMENTS: dict[str, DemoDocument] = {
    d.doc_id: d for d in (NORTHWIND_LOAN, SAHYADRI_KFS, KAVERI_SANCTION, SURAKSHA_HEALTH, SHOPKART_EMI)
}


def list_demo_documents() -> list[dict]:
    """Catalogue for the viewer (JSON-friendly)."""
    return [
        {"id": d.doc_id, "title": d.title, "institution": d.institution, "category": d.category,
         "description": d.description, "default_scanned": d.default_scanned, "pages": len(d.pages),
         "highlights": list(d.highlights)}
        for d in DEMO_DOCUMENTS.values()
    ]
