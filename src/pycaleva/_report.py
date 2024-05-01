"""
This file holds the logic for saving calibration measurement results to reports in pdf format.


References
----------
The report export functionality is using the fpdf2 package.
https://pyfpdf.github.io/fpdf2/index.html

"""

from fpdf import FPDF
from datetime import datetime
from ._basecalib import _BaseCalibrationEvaluator
from ._basecalib import DEVEL
import shutil
import os
import tempfile
import matplotlib.pyplot as plt


class _Report(FPDF):
    def __init__(self):
        super().__init__()
        self.__page_width = None

        # Create a CalibrationEvaluator Instance to get calibration result data from
        self.__ca = None

        # Use temp directory of os to temporarily save plots to
        self.__plot_dir = tempfile.mkdtemp() 
        

    def __header(self, model_name:str) -> None:
        
        # Write title
        self.set_font('Times','B',12.0) 
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.cell(self.__page_width, 0.0, f'Calibration report - {now}', align='C')
        self.ln(8)

        # Write model name
        self.set_font('Courier', '', 10)
        self.cell(self.__page_width/2, 0.0, f"Model: {model_name}", 0, 0, "L")
        self.ln(8)

        # Write type of evaluation
        if self.__ca.outsample == DEVEL.INTERNAL:
            self.cell(self.__page_width/2, 0.0, "Evaluation: Internal", 0, 0, "L")
        else:
            self.cell(self.__page_width/2, 0.0, "Evaluation: External", 0, 0, "L")
        

    def __contingency_table(self) -> None:
        line_height = self.font_size * 2

        df = self.__ca.contingency_table.reset_index(level=0)

        col_width = self.epw / len(df.columns)  # distribute content evenly

        # Write table header
        self.set_font('Courier', 'B', 8)
        for datum in df.columns:
            self.multi_cell(col_width, line_height, datum, border=1, ln=3, max_line_height=self.font_size)
        self.ln(line_height)

        # Write table content
        self.set_font('Courier', '', 8)
        for index,row in df.iterrows():
            for datum in row:
                if (isinstance(datum,float)):
                    self.multi_cell(col_width, line_height, str(round(datum,3)), border=1, ln=3, max_line_height=self.font_size)
                else:
                    self.multi_cell(col_width, line_height, str(datum), border=1, ln=3, max_line_height=self.font_size)
            self.ln(line_height)
        self.ln(8)

    def __stattests(self) -> None:
        hl = self.__ca.hosmerlemeshow(verbose=False);
        if (hl.pvalue < 0.001):
            self.cell(self.__page_width, 0.0, f'Hosmer Lemeshow Test: C({hl.dof})={round(hl.statistic,4)} p-value: < 0.001', align='L')
        else:
            self.cell(self.__page_width, 0.0, f'Hosmer Lemeshow Test: C({hl.dof})={round(hl.statistic,4)} p-value: {round(hl.pvalue,4)}', align='L')
        self.ln(8)

        ph = self.__ca.pigeonheyse(verbose=False);
        if (ph.pvalue < 0.001):
            self.cell(self.__page_width, 0.0, f'Pigeon Heyse Test: J²({ph.dof})={round(ph.statistic,4)} p-value: < 0.001', align='L')
        else:
            self.cell(self.__page_width, 0.0, f'Pigeon Heyse Test: J²({ph.dof})={round(ph.statistic,4)} p-value: {round(ph.pvalue,4)}', align='L')
        self.ln(8)

        zt = self.__ca.z_test()
        if (zt.pvalue < 0.001):
            self.cell(self.__page_width, 0.0, f'Spiegelhalter z-test: Z={round(zt.statistic,4)} p-value: < 0.001', align='L')
        else:
            self.cell(self.__page_width, 0.0, f'Spiegelhalter z-test: Z={round(zt.statistic,4)} p-value: {round(zt.pvalue,4)}', align='L')


        self.ln(8)


    def __create_plots(self):
        # Delete folder if exists and create it again
        try:
            shutil.rmtree(self.__plot_dir)
            os.mkdir(self.__plot_dir)

            fig = self.__ca.calibration_plot()
            fig.savefig(f"{self.__plot_dir}/calplot.png", dpi=300)
            plt.close(fig)

            # TODO: Make parameter devel dynamic
            fig = self.__ca.calbelt(plot=True).fig
            fig.savefig(f"{self.__plot_dir}/calbelt.png", dpi=300)
            plt.close(fig)

        except FileNotFoundError:
            os.mkdir(self.__plot_dir)

    def __plots(self):
        self.__create_plots()

        self.set_font('Courier', 'U', 8)
        self.cell(self.__page_width, 0.0, 'Calibration Plot', align='L')
        self.ln(8)
        self.image(f'{self.__plot_dir}/calplot.png', h=self.eph/2.2, w=self.epw)


        # Add new page
        self.add_page()

        self.set_font('Courier', 'U', 8)
        self.cell(self.__page_width, 0.0, 'Calibration Belt', align='L')
        self.ln(8)
        self.image(f'{self.__plot_dir}/calbelt.png', h=self.eph/2, w=self.epw)
        
        # Clear temporary saved plots
        shutil.rmtree(self.__plot_dir)
        os.mkdir(self.__plot_dir)


    def footer(self):
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        # Setting font: helvetica italic 8
        self.set_font("helvetica", "I", 8)
        # Printing page number:
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def create(self, filepath:str, model_name:str, ca:_BaseCalibrationEvaluator):
        
        self.__ca = ca

        # Add new page
        self.add_page()

        self.__page_width = self.w - 2 * self.l_margin

        self.__header(model_name)   # Write header
        self.ln(8)

        self.__contingency_table()  # Write contingency table
        self.__stattests()          # Write statistic test results
        self.__plots()              # Write plots

        try:
            self.output(filepath, 'F')
            print(f"Calibration report for model '{model_name}' saved to {filepath}")
        except PermissionError:
            raise Exception(f"Could not write report to {filepath} due to permission error. Close the file first!")
        except FileNotFoundError:
            raise Exception(f"Invalid Path for '{filepath}'!")