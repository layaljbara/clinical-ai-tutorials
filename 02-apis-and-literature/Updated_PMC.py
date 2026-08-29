import xml.etree.ElementTree as ET
import html
import re
import json
import os
import csv

# Global cache to prevent re-loading the 10MB SCImago CSV for every article
_SCIMAGO_CACHE = {}
_WARNED_METRICS_MISSING = False

def _load_scimago_csv(csv_path):
    """Loads the SCImago CSV into a high-speed lookup dictionary."""
    global _SCIMAGO_CACHE
    if csv_path in _SCIMAGO_CACHE:
        return _SCIMAGO_CACHE[csv_path]

    if not os.path.exists(csv_path):
        return {}

    db = {}
    try:
        def normalize(issn):
            return re.sub(r'[^0-9X]', '', str(issn).upper())

        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                sjr_val = row.get("SJR", "").replace(",", ".")
                metrics = {
                    "sjr": sjr_val,
                    "quartile": row.get("SJR Best Quartile"),
                    "h_index": row.get("H index"),
                    "rank": row.get("Rank"),
                    "country": row.get("Country"),
                    "region": row.get("Region"),
                    "categories": [c.strip() for c in row.get("Categories", "").split(";")]
                }
                # Map by every ISSN in the row
                for issn in row.get("Issn", "").split(","):
                    norm = normalize(issn)
                    if norm: db[norm] = metrics
                # Map by lowercase title as fallback
                title = row.get("Title", "").lower()
                if title: db[title] = metrics
    except Exception:
        pass # Fail gracefully if CSV is malformed
    
    _SCIMAGO_CACHE[csv_path] = db
    return db

class PMC_article:
    """A class defining attributes and methods for parsing a PMC article XML file"""
    
    BLOCK_TAGS = {"p", "sec", "title", "abstract", "body", "list", "list-item", "table-wrap", "fig", "contrib-group", "aff", "disp-formula", "chem-struct-wrap", "tr", "td", "table", "thead", "tbody", "back", "ack", "ref-list", "ref", "app-group", "app", "glossary", "def-list", "def-item", "notes"}
    INLINE_TAGS = {"italic", "bold", "underline", "sup", "sub", "sc", "ext-link", "inline-formula"}
    REF_MAP = {"table": "TABREF", "fig": "FIGREF", "bibr": "BIBREF", "aff": "AFFREF"}
    DOCUMENT_BOUNDARY = "\n\n<|endoftext|>\n"

    def __init__(self, pmc_xml_path, scimago_csv_path=None, metrics_path=None):
        self.path = pmc_xml_path
        global _WARNED_METRICS_MISSING
        
        # Auto-detect scimago file if it exists in current directory and no path given
        if scimago_csv_path is None:
            if os.path.exists("scimago_latest.csv"):
                self.scimago_csv_path = "scimago_latest.csv"
            else:
                self.scimago_csv_path = None
                if not _WARNED_METRICS_MISSING:
                    print("WARNING: 'scimago_latest.csv' not found. Journal metrics (quartile, SJR, etc.) will be None.")
                    _WARNED_METRICS_MISSING = True
        else:
            if not os.path.exists(scimago_csv_path):
                print(f"WARNING: Scimago CSV not found at '{scimago_csv_path}'.")
            self.scimago_csv_path = scimago_csv_path
            
        self.metrics_path = metrics_path
        self._root = None
        
        # Metadata
        self.article_title = None
        self.journal_name = None
        self.journal_ids = {}
        # Journal Metrics (Direct Simple Attributes)
        self.journal_sjr = None
        self.journal_quartile = None
        self.journal_h_index = None
        self.journal_rank = None
        self.journal_country = None
        self.journal_region = None
        self.journal_categories = []
        
        self.publisher_name = None
        self.publication_date = None
        self.author_names = []
        self.article_type = None
        self.article_categories = []
        self.author_affiliations = []
        
        # Content Strings
        self.abstract_string = None
        self.body_string = None
        self.body_plain_text = None
        self.body_structured_text = None
        self.abstract_plain_text = None
        self.abstract_structured_text = None
        self.back_text = None
        
        # Structured Data
        self.abstract_sections = []
        self.body_sections = []
        
        self._parse_xml()
        self._extract_metadata()
        self._extract_content()



    def _parse_xml(self):
        """Parses XML file and stores the root Element."""
        try:
            tree = ET.parse(self.path)
            self._root = tree.getroot()
        except Exception as e:
            raise RuntimeError(f"Failed to parse XML file: {self.path}") from e

    def _extract_metadata(self):
        """Extract basic article metadata."""
        if self._root is None: return

        # Get actual article element (might be root or a child of articleset)
        article_el = self._root if self._root.tag == "article" else self._root.find(".//article")
        if article_el is not None:
            self.article_type = article_el.get("article-type")

        # 1. Title
        title_el = self._root.find(".//article-title")
        if title_el is not None:
            self.article_title = self.get_smart_text(title_el)

        # 2. Journal
        journal_el = self._root.find(".//journal-title")
        if journal_el is None:
            journal_el = self._root.find(".//source") # fallback
        if journal_el is not None:
            self.journal_name = self.get_smart_text(journal_el)

        # Journal IDs
        j_ids = {}
        for jid_el in self._root.findall(".//journal-id"):
            jid_type = jid_el.get("journal-id-type", "unknown")
            j_ids[jid_type] = jid_el.text
        self.journal_ids = j_ids

        # Publisher
        pub_el = self._root.find(".//publisher-name")
        if pub_el is not None:
            self.publisher_name = pub_el.text

        # 3. Authors & Affiliations
        # First, map all affiliations by ID
        aff_map = {}
        for aff in self._root.findall(".//aff"):
            aff_id = aff.get("id")
            if aff_id:
                # Clean up affiliation text (remove label and machine IDs if present)
                aff_text = self.get_smart_text(aff, ignore_tags={"label", "institution-id"}).strip()
                aff_map[aff_id] = aff_text

        authors = []
        author_affs = []
        for contrib in self._root.findall(".//contrib[@contrib-type='author']"):
            name = contrib.find("name")
            if name is not None:
                surname = name.find("surname")
                given = name.find("given-names")
                s = surname.text if surname is not None else ""
                g = given.text if given is not None else ""
                full_name = f"{g} {s}".strip()
                if full_name:
                    authors.append(full_name)
                    
                    # Get specific affiliations for this author
                    my_affs = []
                    for xref in contrib.findall("xref[@ref-type='aff']"):
                        rid = xref.get("rid")
                        if rid in aff_map:
                            my_affs.append(aff_map[rid])
                    
                    author_affs.append({
                        "name": full_name,
                        "affiliations": my_affs
                    })
        self.author_names = authors
        self.author_affiliations = author_affs

        # Categories
        categories = []
        for subj_el in self._root.findall(".//subj-group/subject"):
            if subj_el.text:
                categories.append(subj_el.text)
        self.article_categories = sorted(list(set(categories)))

        # ISSNs (for metrics lookup)
        for issn_el in self._root.findall(".//issn"):
            issn_type = issn_el.get("pub-type", "unknown")
            self.journal_ids[f"ISSN_{issn_type}"] = issn_el.text

        # 4. Publication Date
        # Prefer epub, then ppub, then anything with a year
        pub_dates = self._root.findall(".//pub-date")
        best_date = None
        for pd in pub_dates:
            p_type = pd.get("pub-type")
            year = pd.find("year")
            if year is not None:
                month = pd.find("month")
                day = pd.find("day")
                y = year.text
                m = month.text if month is not None else "01"
                d = day.text if day is not None else "01"
                date_str = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                if p_type == "epub":
                    best_date = date_str
                    break # Epub is usually most specific for PMC
                elif p_type == "ppub" or not best_date:
                    best_date = date_str
        self.publication_date = best_date

        # Lookup Journal Metrics (On-the-fly)
        if self.scimago_csv_path or self.metrics_path:
            self._lookup_journal_metrics()

    def _lookup_journal_metrics(self):
        """Lookup impact factor and quartile on-the-fly."""
        db = {}
        # 1. Try Scimago CSV cache FIRST (High performance on-the-fly)
        if self.scimago_csv_path:
            db.update(_load_scimago_csv(self.scimago_csv_path))
            
        # 2. Try JSON metrics file (legacy/manual override)
        if self.metrics_path and os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    db.update(json.load(f))
            except Exception: pass

        if not db: return

        try:
            def normalize(issn):
                return re.sub(r'[^0-9X]', '', str(issn).upper())
                
            # Try matching by ISSNs
            search_keys = [normalize(v) for k, v in self.journal_ids.items() if k.startswith("ISSN")]
            # Try matching by Journal Name (normalized)
            if self.journal_name:
                search_keys.append(self.journal_name.lower())
                
            for key in search_keys:
                if key in db:
                    m = db[key]
                    self.journal_sjr = m.get("sjr")
                    self.journal_quartile = m.get("quartile")
                    self.journal_h_index = m.get("h_index")
                    self.journal_rank = m.get("rank")
                    self.journal_country = m.get("country")
                    self.journal_region = m.get("region")
                    self.journal_categories = m.get("categories", [])
                    break
        except Exception:
            pass # Silent fail for metrics lookup

    def _extract_content(self):
        """Extract abstract and body content."""
        # Abstract
        abstract_el = self._root.find(".//abstract")
        if abstract_el is not None:
            self.abstract_string = ET.tostring(abstract_el, encoding="unicode", method="xml")
        if abstract_el is not None:
            self.abstract_string = ET.tostring(abstract_el, encoding="unicode", method="xml")
            
            # Abstract variants
            self.abstract_plain_text = self.get_smart_text(abstract_el, ignore_tags={"title"})
            self.abstract_structured_text = self.get_smart_text(abstract_el, separator="\n")

            # Structured sections
            self.abstract_sections = self._parse_sections(abstract_el)
            if not self.abstract_sections:
                # Flat abstract - treat as one section
                self.abstract_sections = [{
                    "type": "section",
                    "title": "Abstract",
                    "text": self.abstract_plain_text,
                    "subsections": []
                }]

        # Body
        body_el = self._root.find(".//body")
        
        if body_el is not None:
            self.body_string = ET.tostring(body_el, encoding="unicode", method="xml")
            self.body_plain_text = self.get_smart_text(body_el, ignore_tags={"title"})
            
            # Hierarchical extraction for structured data
            self.body_sections = self._parse_sections(body_el)

        # Final structured text for the whole article

        # Final structured text for the whole article
        self.body_structured_text = self._generate_body_structured_text()

    def _generate_body_structured_text(self):
        """
        Actually generates the structured text representation.
        """
        parts = []
        
        # Body only (User request)
        body_el = self._root.find(".//body")
        if body_el is not None:
            parts.append(self.get_smart_text(body_el, separator="\n"))

        # Join major sections with double newlines
        full_text = "\n\n".join(p for p in parts if p.strip())
        
        return full_text + self.DOCUMENT_BOUNDARY

    def _parse_sections(self, root_el):
        """Hierarchical list of top-level sections."""
        sections = []
        for child in root_el:
            if child.tag == "title":
                continue # Skip the container title
            if child.tag == "sec":
                sections.append(self._parse_section_recursive(child))
            elif child.tag in self.BLOCK_TAGS or child.tag == "p":
                text = self.get_smart_text(child)
                if text:
                    sections.append({
                        "type": "paragraph",
                        "text": text
                    })
        return sections

    def _parse_section_recursive(self, sec_el):
        """Recursively parse a single section."""
        section = {
            "type": "section",
            "title": None,
            "text": "",
            "subsections": []
        }
        
        # Get title
        title_el = sec_el.find("title")
        if title_el is not None:
            section["title"] = self.get_smart_text(title_el)
            
        # Get direct child paragraphs (exclude those in subsections)
        p_texts = []
        for child in sec_el:
            if child.tag == "p":
                p_texts.append(self.get_smart_text(child))
            elif child.tag == "sec":
                section["subsections"].append(self._parse_section_recursive(child))
            elif child.tag in self.BLOCK_TAGS:
                # Other blocks like lists/tables
                text = self.get_smart_text(child)
                if text:
                    p_texts.append(text)
                
        section["text"] = " ".join(p_texts)
        return section

    @classmethod
    def _extract_equation(cls, formula_el):
        """
        Extract equation content from a formula element.
        Tries to get TeX content first, then MathML, then plain text.
        """
        # Try to find tex-math element
        tex_math = formula_el.find(".//tex-math")
        if tex_math is not None and tex_math.text:
            return tex_math.text.strip()
        
        # Try to find mml:math element (MathML)
        # Note: MathML namespace might be present
        mml_math = formula_el.find(".//{http://www.w3.org/1998/Math/MathML}math")
        if mml_math is None:
            mml_math = formula_el.find(".//math")
        
        if mml_math is not None:
            # For MathML, we'll extract the text content as a fallback
            # A full MathML to LaTeX converter would be complex
            mathml_text = ET.tostring(mml_math, encoding='unicode', method='text').strip()
            if mathml_text:
                return f"[MathML: {mathml_text}]"
        
        # Fallback: extract all text content using itertext to avoid infinite recursion
        text_content = "".join(formula_el.itertext())
        return text_content.strip() if text_content else "equation"

    @staticmethod
    def _is_simple_formula(text):
        """Heuristic to decide if a formula is just a simple variable/symbol."""
        if not text: return False
        # Remove common math symbols and see if it's very short
        # e.g. 'S', 'I', 't', 'Δt', '&#946;' (beta), 'R0'
        clean = text.replace('&#946;', 'b').replace('&#916;', 'D').strip()
        if len(text) <= 3:
            return True
        # Also handle common things like 'Itot' or 't+Dt' if they are short
        if len(text) <= 5 and not any(c in text for c in {'=', '>', '<', '{', '\\'}):
            return True
        return False

    @staticmethod
    def print_tree_structure(element, level=1):
        """Prints tree structure"""
        for child in element:
            print(" "*level, "-", child.tag)
            self.print_tree_structure(element=child,level=level+1)

    
    @classmethod
    def get_smart_text(cls, element, ignore_tags=None, separator=" "):
        """
        Recursively extract text from an element with smart spacing and reference markers.
        Allows ignoring specific tags (e.g. 'title' to get plain text without headings).
        'separator' controls what is placed around block tags (" " or "\n").
        """
        if element is None:
            return ""

        ignore_tags = ignore_tags or set()
        newline_mode = (separator == "\n")

        def _walk(el, is_root=False, parent_tag=None):
            tag = el.tag
            
            # Skip entire subtree if tag is ignored
            if tag in ignore_tags:
                return ""

            # Mathematical Equation Formatting
            if tag == "disp-formula":
                return " <MATH_FORMULA> "
            
            if tag == "inline-formula":
                # User preference: keep simple variables (S, I, R, t, etc.) as text
                formula_text = "".join(el.itertext()).strip()
                if cls._is_simple_formula(formula_text):
                    return formula_text
                return "<MATH_FORMULA>"

            # XREF Handling: Replace completely with token (Numbers = Noise)
            if tag == "xref":
                ref_type = el.get("ref-type")
                if ref_type in cls.REF_MAP:
                    return f"<{cls.REF_MAP[ref_type]}>"
                return ""

            # Table, Figure, and Supporting Information Handling: Skip completely.
            if tag in {"table-wrap", "fig", "supplementary-material", "inline-supplementary-material"}:
                return ""
            
            # Skip Supporting Information sections specifically
            if tag == "sec" and el.get("sec-type") == "supplementary-material":
                return ""

            parts = []
            
            # Space/Newline before block elements
            is_block = tag in cls.BLOCK_TAGS
            # Special case: don't start a new line for a paragraph if it's inside a list-item
            # because we want "1. Text" format.
            if tag == "p" and parent_tag == "list-item":
                is_block = False

            if is_block:
                parts.append(separator)
            elif tag == "p":
                parts.append(" ") # Formatted paragraph in list-item gets a space

            # Pre-text (text before first child)
            if el.text:
                parts.append(el.text)
            
            # Recursive walk through children
            for child in el:
                parts.append(_walk(child, is_root=False, parent_tag=tag))
                if child.tail:
                    parts.append(child.tail)
            
            # Special handling for ext-link (Append URL)
            if tag == "ext-link":
                url = el.get("{http://www.w3.org/1999/xlink}href") or el.get("href")
                if url:
                    parts.append(f" ({url})")

            # Space/Newline after block elements
            if is_block:
                parts.append(separator)

            return "".join(parts)

        raw_text = _walk(element, is_root=True)
        return cls.clean_for_llm(raw_text, preserve_newlines=newline_mode)

    @classmethod
    def clean_for_llm(cls, text, preserve_newlines=False):
        """
        Advanced cleaning for LLM training:
        - Decodes entities
        - Normalizes whitespace (optionally preserving newlines)
        - Fixes punctuation spacing
        - Tidies up reference markers
        """
        if not text:
            return ""

        # 1. Decode HTML entities
        text = html.unescape(text)

        # 2. Normalize whitespace
        if preserve_newlines:
            # Collapse horizontal tabs/multiple spaces but keep \n
            text = re.sub(r'[ \t\r]+', ' ', text)
        else:
            # Collapse all whitespace to single space
            text = re.sub(r'\s+', ' ', text)

        # 3. Punctuation spacing cleanup
        text = re.sub(r' +([.,;?!])', r'\1', text)
        # Only add space after punctuation if it's NOT likely a URL or abbreviation
        # Rule: Add space after punctuation if preceded by whitespace, OR if it's lowercase followed by Uppercase.
        text = re.sub(r'(?<= )([.,;?!])([^\s\d\]\)\.,;?!])', r'\1 \2', text)
        text = re.sub(r'([a-z])([.,;?!])([A-Z])', r'\1\2 \3', text)
        
        # 4. Abbreviation protection & joining
        text = re.sub(r'\b([A-Za-z])\.\s+(?=([A-Za-z]\.))', r'\1.', text)
        text = re.sub(r'\bet\s+al\s*\.', 'et al.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bvs\s*\.', 'vs.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bapprox\s*\.', 'approx.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bno\s*\.', 'no.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bvol\s*\.', 'vol.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bref\s*\.', 'ref.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcf\s*\.', 'cf.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bviz\s*\.', 'viz.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bca\s*\.', 'ca.', text, flags=re.IGNORECASE)
        text = re.sub(r'\bInc\s*\.', 'Inc.', text, flags=re.IGNORECASE)

        # 4b. Citation Prefix Cleanup (Strip "Table 1", "Figure 2" before tokens)
        # e.g. "Table 1 <TABREF>" -> "<TABREF>"
        # e.g. "Fig. 2 <FIGREF>" -> "<FIGREF>"
        # e.g. "shown in Table 1 <TABREF>" -> "shown in <TABREF>"
        
        # Regex explanation:
        # (Table|Tables|Figure|Figures|Fig\.?|Figs\.?): Match keywords
        # \s*: Optional space
        # [\d\w\-\.,]*: Match numbers/identifiers (1, 1a, 1-3, etc)
        # \s*: Optional space
        # (?=<[A-Z]+REF>): Lookahead for the token
        prefix_pattern = r'\b(Table|Tables|Figure|Figures|Fig\.?|Figs\.?)\s*[\d\w\-\.,]*\s*(?=<[A-Z]+REF>)'
        text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE)

        # 4c. Citation Suffix Cleanup (Strip "a", "b", "c" etc. after tokens)
        # e.g. "<FIGREF> a" -> "<FIGREF>"
        # e.g. "(<FIGREF> b)" -> "(<FIGREF>)" -> "<FIGREF>" (brackets handled later)
        for marker in cls.REF_MAP.values():
            m_tag = f"<{marker}>"
            m = re.escape(m_tag)
            # Remove trailing space + single char label (a, b, c...) or small number
            # only if followed by punctuation, closing bracket, or end of string.
            text = re.sub(rf'({m})\s+[a-zA-Z0-9]\b(?=[\s\)\],.;!?]|$)', r'\1', text)
        
        # 5. Reference marker tidying & Collapsing
        for marker in cls.REF_MAP.values():
            m_tag = f"<{marker}>"
            m = re.escape(m_tag)
            
            # Collapse adjacent tokens separated by spaces, commas, semicolons, dashes
            # e.g., <BIBREF>, <BIBREF> -> <BIBREF>
            # e.g., <BIBREF>-<BIBREF> -> <BIBREF>
            # e.g., <BIBREF><BIBREF> -> <BIBREF> (Strict adjacency)
            text = re.sub(rf'{m}(?:\s*[,;–-]\s*{m})+', m_tag, text)
            text = re.sub(rf'{m}(?:\s*{m})+', m_tag, text)
            
            # Cleanup binding to previous words/brackets
            # Remove brackets wrapping a token: [<BIBREF>] -> <BIBREF>
            text = re.sub(r'\[\s*' + m + r'\s*\]', m_tag, text)
            text = re.sub(r'\(\s*' + m + r'\s*\)', m_tag, text)
            
            # Ensure space before token if it follows a word (unless it's at start)
            text = re.sub(rf'([a-zA-Z0-9])({m})', r'\1 \2', text)
            
            # Remove alphanumeric suffixes after tokens (e.g., <FIGREF>a -> <FIGREF>)
            text = re.sub(rf'{m}[a-zA-Z0-9]+', m_tag, text)
            
            # Ensure punctuation follows token immediately if present
            text = re.sub(rf'{m}\s+([.,;:!?])', rf'{m_tag}\1', text)

        # 6. Strip empty brackets that might have contained only removed numbers
        # e.g. "Previous work [ ]." -> "Previous work ."
        # Apply multiple times for nested or messy brackets: ([ <FIGREF> ])
        for _ in range(2):
            text = re.sub(r'\[\s*\]', '', text)
            text = re.sub(r'\(\s*\)', '', text)
            # Also strip brackets mapping ONLY to a token (with possible spaces)
            for marker in cls.REF_MAP.values():
                m = re.escape(f"<{marker}>")
                text = re.sub(r'\[\s*' + m + r'\s*\]', f"<{marker}>", text)
                text = re.sub(r'\(\s*' + m + r'\s*\)', f"<{marker}>", text)
            
        # 7. Final whitespace strip
        if preserve_newlines:
            # Collapse multiple spaces
            text = re.sub(r' +', ' ', text)
            # Collapse triple+ newlines to double
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            # Remove spaces at start/end of lines
            text = "\n".join(line.strip() for line in text.split("\n"))
            text = text.strip()
        else:
            text = re.sub(r'\s+', ' ', text).strip()

        return text



# --- Test Section ---
if __name__ == "__main__":
    # Create a dummy XML file for testing
    test_xml = """
    <article>
        <front>
            <journal-title>Journal of Testing</journal-title>
            <article-title>Effect of <sub>X</sub> on <sup>Y</sup></article-title>
            <abstract>
                <sec>
                    <title>Methods</title>
                    <p>We looked at Table 1 <xref ref-type="table">1</xref> and Figure 2 <xref ref-type="fig">2</xref>.</p>
                    <p>Referencing Smith et al. <xref ref-type="bibr">23</xref>.</p>
                </sec>
                <sec>
                    <title>Results</title>
                    <p>Concentration (C<sub>max</sub>) was high.</p>
                </sec>
            </abstract>
        </front>
    </article>
    """
    with open("test_pmc.xml", "w") as f:
        f.write(test_xml)
        
    article = PMC_article("test_pmc.xml")
    print(f"Title: {article.article_title}")
    print(f"Abstract Full: {article.abstract_full_text}")
    print(f"Abstract Plain: {article.abstract_plain_text}")
    for sec in article.abstract_sections:
        print(f"Section [{sec['title']}]: {sec['text']}")
