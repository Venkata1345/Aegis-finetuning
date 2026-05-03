"""ai4privacy label taxonomy -> Aegis 9-category mapping.

The source dataset has ~50+ fine-grained labels (e.g. FIRSTNAME, LASTNAME1,
BUILDINGNUMBER, IPV4). We map them down to nine HIPAA+GDPR-aligned types.

Design notes:
- **Confidence-scoped mapping.** Only labels we're confident about are
  seeded. Any source label not in the mapping is reported by the converter
  (with occurrence counts) so we update this file empirically rather than
  guessing labels that may not exist in the data.
- **DATE deliberately does NOT map to DOB.** A bare DATE could be anything;
  the adversarial test set explicitly checks that the model doesn't flag
  arbitrary dates as DOBs. Only DOB / DATEOFBIRTH / BIRTHDATE map to DOB.
- **USERNAME -> PERSON.** A username links to a specific individual and is
  in scope for HIPAA/GDPR PII.
- Source-label keys are upper-cased before lookup; lookup is case-insensitive.
"""

from __future__ import annotations

from data.schema import PIIType

_AI4P_TO_AEGIS: dict[str, PIIType] = {
    # PERSON
    "FIRSTNAME": "PERSON",
    "LASTNAME": "PERSON",
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",
    "MIDDLENAME": "PERSON",
    "GIVENNAME": "PERSON",
    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",
    "FULLNAME": "PERSON",
    "NAME": "PERSON",
    "PREFIX": "PERSON",
    "SUFFIX": "PERSON",
    "TITLE": "PERSON",
    "USERNAME": "PERSON",
    # EMAIL
    "EMAIL": "EMAIL",
    "EMAILADDRESS": "EMAIL",
    # PHONE
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONENUM": "PHONE",
    "TEL": "PHONE",
    "TELEPHONE": "PHONE",
    "TELEPHONENUM": "PHONE",
    "TELEPHONENUMBER": "PHONE",
    # ADDRESS — all physical-address components collapse to one category
    "ADDRESS": "ADDRESS",
    "STREET": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "STREETADDRESS1": "ADDRESS",
    "STREETADDRESS2": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "BUILDING": "ADDRESS",
    "BUILDINGNAME": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "SECADDRESS": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "COUNTY": "ADDRESS",
    "COUNTRY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "POSTCODE": "ADDRESS",
    "GEOCOORD": "ADDRESS",
    # DOB — only labels that explicitly mean date-of-birth.
    # BOD is ai4privacy's abbreviation for date-of-birth; bare DATE is intentionally NOT
    # mapped (the adversarial set explicitly checks "a date that's NOT a DOB").
    "DOB": "DOB",
    "DATEOFBIRTH": "DOB",
    "BIRTHDATE": "DOB",
    "BIRTHDAY": "DOB",
    "BOD": "DOB",
    # GOV_ID
    "SSN": "GOV_ID",
    "SOCIALNUMBER": "GOV_ID",
    "SOCIALSECURITYNUMBER": "GOV_ID",
    "NATIONALID": "GOV_ID",
    "PASSPORT": "GOV_ID",
    "PASSPORTNUM": "GOV_ID",
    "PASSPORTNUMBER": "GOV_ID",
    "DRIVERLICENSE": "GOV_ID",
    "DRIVERLICENSENUM": "GOV_ID",
    "IDCARD": "GOV_ID",
    "IDNUM": "GOV_ID",
    "TAXNUM": "GOV_ID",
    # FINANCIAL
    "CREDITCARDNUMBER": "FINANCIAL",
    "CREDITCARD": "FINANCIAL",
    "CREDITCARDCVV": "FINANCIAL",
    "BANKACCOUNT": "FINANCIAL",
    "BANKACCOUNTNUM": "FINANCIAL",
    "ACCOUNTNUM": "FINANCIAL",
    "ACCOUNTNUMBER": "FINANCIAL",
    "IBAN": "FINANCIAL",
    "BIC": "FINANCIAL",
    "ROUTINGNUMBER": "FINANCIAL",
    "BITCOINADDRESS": "FINANCIAL",
    "ETHEREUMADDRESS": "FINANCIAL",
    "LITECOINADDRESS": "FINANCIAL",
    # MEDICAL_ID — likely sparse in ai4privacy; included for completeness
    "HEALTHCAREID": "MEDICAL_ID",
    "MEDICALRECORDNUMBER": "MEDICAL_ID",
    "MRN": "MEDICAL_ID",
    "HEALTHPLANID": "MEDICAL_ID",
    # IP_ADDRESS
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
}


def map_label(src: str | None) -> PIIType | None:
    """Map an ai4privacy label to an Aegis category, or None to skip."""
    if not src:
        return None
    return _AI4P_TO_AEGIS.get(src.upper())


def known_labels() -> set[str]:
    return set(_AI4P_TO_AEGIS)
