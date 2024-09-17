from fpdf import FPDF

class ReportGenerator:
    def __init__(self, report_data):
        self.report_data = report_data

    def export(self, format='html'):
        """Exporta el informe en el formato indicado."""
        if format == 'html':
            return self.generate_html_report()
        elif format == 'pdf':
            return self.generate_pdf_report()
        else:
            raise ValueError("Formato no soportado: usa 'html' o 'pdf'.")

    def generate_html_report(self):
        """Genera un informe en formato HTML."""
        html_report = "<html><body><h1>Informe de Datos</h1>"
        for key, value in self.report_data.items():
            html_report += f"<h2>{key}</h2><pre>{value}</pre>"
        html_report += "</body></html>"
        return html_report

    def generate_pdf_report(self):
        """Genera un informe en formato PDF."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Informe de Datos", ln=True, align="C")
        for key, value in self.report_data.items():
            pdf.cell(200, 10, txt=f"{key}: {str(value)}", ln=True, align="L")
        return pdf.output("reporte.pdf")
