import os
import ROOT
from ROOT import TCanvas, TPad, TLegend, TLatex
from ROOT import THStack
import tdrstyle; tdrstyle.setTDRStyle(square=False)
ROOT.gStyle.SetErrorX(0.5)
ROOT.gStyle.SetAxisMaxDigits(3)

PeriodInfo = {
        "2016preVFP": ["B_ver2", "C", "D", "E", "F"],
        "2016postVFP": ["F", "G", "H"],
        "2017": ["B", "C", "D", "E", "F"],
        "2018": ["A", "B", "C", "D"]
}

LumiInfo = {    # /fb
        "2016preVFP": 19.5,
        "2016postVFP": 16.8,
        "2017": 41.5,
        "2018": 59.8
}

class KinematicCanvas():
    def __init__(self, config):
        self.config = config
        
        # initialize default settings
        self.cvs = TCanvas("c", "", 1600, 1500)
        self.cvs.SetLeftMargin(0.1)
        self.cvs.SetRightMargin(0.08)
        self.cvs.SetTopMargin(0.06)
        self.cvs.SetBottomMargin(0.1)
        self.lumi = TLatex()
        self.lumi.SetTextSize(0.032)
        self.lumi.SetTextFont(42)
        self.cms = TLatex()
        self.cms.SetTextSize(0.035)
        self.cms.SetTextFont(61)
        self.preliminary = TLatex()
        self.preliminary.SetTextSize(0.032)
        self.preliminary.SetTextFont(52)
        
        self.signals = None
        self.backgrounds = None
        self.stack = THStack("stack", "")
        self.systematics = None
        
        # optional settings
        self.logy = False
        if "logy" in config.keys():
            self.logy = config['logy']
        self.lumiString = ""
        if "era" in config.keys():
            era = config['era']
            self.lumiString = "L_{int} ="+f" {LumiInfo[era]}"+" fb^{-1} (13TeV)"
    
    def drawSignals(self, hists, colors):
        self.signals = hists
        
        # rebin
        if "rebin" in self.config.keys():
            for hist in self.signals.values(): hist.Rebin(self.config['rebin'])
        
        # color
        for hist in self.signals.values():
            hist.SetStats(0)
            color = colors[hist.GetName()]
            hist.SetLineColor(color)
            hist.SetLineWidth(4)
            hist.SetFillColorAlpha(color, 0.2)

        # X axis
        xRange = None
        if "xRange" in self.config.keys():
            xRange = self.config['xRange']
        for hist in self.signals.values():
            hist.GetXaxis().SetTitle(self.config['xTitle'])
            hist.GetXaxis().SetTitleSize(0.04)
            hist.GetXaxis().SetTitleOffset(1.0)
            hist.GetXaxis().SetLabelSize(0.035)
            if xRange:
                hist.GetXaxis().SetRangeUser(xRange[0], xRange[1])
        
        # Y axis
        maximum = max([h.GetMaximum() for h in self.signals.values()])
        for hist in self.signals.values():
            hist.GetYaxis().SetTitle(self.config['yTitle'])
            hist.GetYaxis().SetRangeUser(0, maximum*2.)
            
    def drawBackgrounds(self, hists, colors):
        self.backgrounds = hists
        
        # rebin
        if "rebin" in self.config.keys():
            for hist in self.backgrounds.values(): hist.Rebin(self.config['rebin']) 
        
        # color
        for hist in self.backgrounds.values():
            hist.SetStats(0)
            color = colors[hist.GetName()]
            hist.SetFillColorAlpha(color, 0.5)

        for hist in self.backgrounds.values():
            self.stack.Add(hist)
            if self.systematics is None: self.systematics = hist.Clone("syst")
            else:                        self.systematics.Add(hist)
        self.stack.Draw()   # to use self.stack.GetHistogram()
        
        # X axis
        xRange = None
        if "xRange" in self.config.keys():
            xRange = self.config['xRange']
        
        self.stack.GetHistogram().SetTitle(self.config['xTitle'])
        self.stack.GetHistogram().GetXaxis().SetTitleOffset(1.0)
        self.stack.GetHistogram().GetXaxis().SetTitleSize(0.04)
        self.stack.GetHistogram().GetXaxis().SetLabelSize(0.035)
        if xRange: self.stack.GetHistogram().GetXaxis().SetRangeUser(xRange[0], xRange[1])
        
        # y axis
        self.stack.GetHistogram().GetYaxis().SetTitle(self.config['yTitle'])
        self.stack.GetHistogram().GetYaxis().SetTitleOffset(1.0)
        self.stack.GetHistogram().GetYaxis().SetTitleSize(0.04)
        self.stack.GetHistogram().GetYaxis().SetLabelSize(0.04)

        maximum = max([h.GetMaximum() for h in self.signals.values()] +[self.stack.GetHistogram().GetMaximum()])
        if self.logy:
            self.stack.GetHistogram().GetYaxis().SetRangeUser(0.5, maximum*500.)
            for hist in self.signals.values():
                hist.GetYaxis().SetRangeUser(0.5, maximum*500.)
        else:
            self.stack.GetHistogram().GetYaxis().SetRangeUser(0., maximum*2.)
            for hist in self.signals.values():
                hist.GetYaxis().SetRangeUser(0., maximum*2.)
            
        self.systematics.SetStats(0)
        self.systematics.SetFillColorAlpha(12, 0.99)
        self.systematics.SetFillStyle(3144)
        self.systematics.GetXaxis().SetLabelSize(0)
        
    def drawLegend(self):
        self.sigLegend = None
        self.bkgLegend = None
        if self.signals:
            self.sigLegend = TLegend(0.65, 0.6, 0.9, 0.85)
            #self.sigLegend = TLegend(0.35, 0.6, 0.6, 0.85)
            self.sigLegend.SetFillStyle(0)
            self.sigLegend.SetBorderSize(0)
            for hist in self.signals.values():
                self.sigLegend.AddEntry(hist, hist.GetName(), "f")
        
        if self.backgrounds:
            self.bkgLegend = TLegend(0.65, 0.6, 0.9, 0.85)
            self.bkgLegend.SetFillStyle(0)
            self.bkgLegend.SetBorderSize(0)
            for hist in list(self.backgrounds.values())[::-1]:
                self.bkgLegend.AddEntry(hist, hist.GetName(), "f")
        
        if self.systematics:
            self.bkgLegend.AddEntry(self.systematics, "stat+syst", "f")

        # Relocate self.sigLegend to the left
        if self.sigLegend and self.bkgLegend:
            self.sigLegend.SetX1(0.35)
            self.sigLegend.SetX2(0.6)   
        
    def finalize(self):
        if self.logy: self.cvs.SetLogy()
        self.cvs.cd()

        if self.backgrounds:
            self.stack.Draw("hist&same")
        if self.systematics:
            self.systematics.Draw("e2&f&same")
        if self.signals:
            for hist in self.signals.values():
                hist.Draw("f&hist&same")
        
        if self.sigLegend:
            self.sigLegend.Draw()
        if self.bkgLegend:
            self.bkgLegend.Draw()
        
        self.lumi.DrawLatexNDC(0.66, 0.953, self.lumiString)
        self.cms.DrawLatexNDC(0.105, 0.953, "CMS")
        self.preliminary.DrawLatexNDC(0.19, 0.953, "Preliminary") 
        self.cvs.RedrawAxis()
        
    def draw(self):
        self.cvs.Draw()
        
    def SaveAs(self, name):
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.cvs.SaveAs(name)


class ComparisonCanvas():
    def __init__(self, config, name="c", title=""):
        self.config = config
        
        # initialize default settings
        self.cvs = TCanvas(name, title, 820, 900)
        self.padUp = TPad("up", "", 0, 0.25, 1, 1)
        self.padUp.SetTopMargin(0.05)
        self.padUp.SetBottomMargin(0.01)
        self.padUp.SetLeftMargin(0.115)
        self.padUp.SetRightMargin(0.08)
        self.padDown = TPad("down", "", 0, 0, 1, 0.25)
        self.padDown.SetGrid(True)
        self.padDown.SetTopMargin(0.01)
        self.padDown.SetBottomMargin(0.25)
        self.padDown.SetLeftMargin(0.115)
        self.padDown.SetRightMargin(0.08)
        
        self.lumi = TLatex()
        self.lumi.SetTextSize(0.035)
        self.lumi.SetTextFont(42)
        self.cms = TLatex()
        self.cms.SetTextSize(0.04)
        self.cms.SetTextFont(61)
        self.preliminary = TLatex()
        self.preliminary.SetTextSize(0.035)
        self.preliminary.SetTextFont(52)
        
        self.data = None
        self.signals = None
        self.backgrounds = None
        self.stack = THStack("stack", "")
        self.systematics = None
        self.ratio = None
        self.ratio_syst = None
        self.legend = TLegend(0.6, 0.45, 0.85, 0.85)
        self.legend.SetFillStyle(0)
        self.legend.SetBorderSize(0)
        
        self.maximum = -1.
        
        # optional settings
        self.logy = False
        if "logy" in config.keys():
            self.logy = config['logy']
        if "era" in config.keys():
            era = config['era']
            if "lumiInfo" in config.keys():
                self.lumiString = config["lumiInfo"]
            else:
                self.lumiString = "L^{int} = "+f"{LumiInfo[era]}"+" fb^{-1} (13TeV)"
    
    def drawSignals(self, hists, colors):
        self.signals = hists
        
        # Reset legend
        self.legend = TLegend(0.65, 0.6, 0.9, 0.85)
        self.legend.SetFillStyle(0)
        self.legend.SetBorderSize(0)
        self.sigLegend = TLegend(0.35, 0.6, 0.6, 0.85)
        self.sigLegend.SetFillStyle(0)
        self.sigLegend.SetBorderSize(0)
        
        # rebin
        if "rebin" in self.config.keys():
            for hist in self.signals.values(): hist.Rebin(self.config['rebin'])
        
        # color
        for hist in self.signals.values():
            hist.SetStats(0)
            color = colors[hist.GetName()]
            hist.SetLineColor(color)
            hist.SetLineWidth(2)
            hist.SetMarkerColor(color)

        # only one signal i.e. score
        if len(self.signals.values()) == 1:
            hist = list(self.signals.values())[0]
            hist.SetFillColorAlpha(colors[hist.GetName()], 0.4)

        # X axis
        xRange = None
        if "xRange" in self.config.keys():
            xRange = self.config['xRange']
        for hist in self.signals.values():
            hist.GetXaxis().SetTitle(self.config['xTitle'])
            hist.GetXaxis().SetTitleSize(0.04)
            hist.GetXaxis().SetTitleOffset(1.0)
            hist.GetXaxis().SetLabelSize(0.04)
            if xRange: hist.GetXaxis().SetRangeUser(xRange[0], xRange[1])
            self.maximum = max(self.maximum, hist.GetMaximum())

    def drawBackgrounds(self, hists, colors):
        self.backgrounds = hists
        
        # rebin
        if "rebin" in self.config.keys():
            for hist in self.backgrounds.values(): hist.Rebin(self.config['rebin']) 
            
        # color
        for hist in self.backgrounds.values():
            hist.SetStats(0)
            color = colors[hist.GetName()]
            hist.SetFillColorAlpha(color, 0.75)
        
        # make a stack
        for hist in self.backgrounds.values():
            self.stack.Add(hist)
            if self.systematics == None: self.systematics = hist.Clone("syst")
            else:                        self.systematics.Add(hist)
        self.stack.Draw()
        self.stack.GetHistogram().GetXaxis().SetLabelSize(0)
        self.systematics.SetFillColorAlpha(ROOT.kBlack, 0.7)
        self.systematics.SetFillStyle(3144)
        self.systematics.GetXaxis().SetLabelSize(0)
        
        # x axis
        xRange = None
        if "xRange" in self.config.keys():
            xRange = self.config['xRange']
        for hist in self.backgrounds.values():
            hist.GetXaxis().SetLabelSize(0)
            if xRange: hist.GetXaxis().SetRangeUser(xRange[0], xRange[1])

        # y axis
        self.maximum = max(self.maximum, self.stack.GetHistogram().GetMaximum())
        for hist in self.backgrounds.values():
            if self.logy: hist.GetYaxis().SetRangeUser(0.1, self.maximum*1000.)
            else:         hist.GetYaxis().SetRangeUser(0, self.maximum*2.)
    
    def drawData(self, data):
        self.data = data
        if "rebin" in self.config.keys():
            data.Rebin(self.config['rebin'])
        
        self.data.SetStats(0)
        self.data.SetTitle("")
        
        # x axis
        xRange = None
        if "xRange" in self.config.keys():
            xRange = self.config['xRange']
        if xRange: self.data.GetXaxis().SetRangeUser(xRange[0], xRange[1])
        self.data.GetXaxis().SetLabelSize(0)
        self.data.GetYaxis().SetTitleOffset(1.5)
        self.data.GetYaxis().SetLabelSize(0.03)
        if "yTitle" in self.config.keys(): 
            self.data.GetYaxis().SetTitle(self.config["yTitle"])
        self.data.SetMarkerStyle(8)
        self.data.SetMarkerSize(1)
        self.data.SetMarkerColor(1)
        
        #maximum = self.stack.GetHistogram().GetMaximum() 
        if self.logy:
            minValue = 1.
            for i in range(1, self.data.GetNbinsX()+1):
                if self.data.GetBinContent(i) > 0 and self.data.GetBinContent(i) < minValue:
                    minValue = self.data.GetBinContent(i)
            self.data.GetYaxis().SetRangeUser(minValue*0.1, self.maximum*1000.)
        else:         
            self.data.GetYaxis().SetRangeUser(0, self.maximum*2.)
        
        
    def drawRatio(self):
        self.ratio = self.data.Clone("ratio")
        self.ratio.Divide(self.systematics)
        self.ratioSyst = self.ratio.Clone("ratioSyst")
        
        self.ratio.SetStats(0)
        self.ratio.SetTitle("")

        # x axis
        self.ratio.GetXaxis().SetTitle(self.config['xTitle'])
        self.ratio.GetXaxis().SetTitleSize(0.1)
        self.ratio.GetXaxis().SetTitleOffset(0.9)
        self.ratio.GetXaxis().SetLabelSize(0.08)

        # y axis
        self.ratio.GetYaxis().SetRangeUser(0.5, 1.5)
        if "yRange" in self.config.keys():
            self.ratio.GetYaxis().SetRangeUser(self.config['yRange'][0], self.config['yRange'][1])
        if "ratio" in self.config.keys():
            yDown, yUp = self.config['ratio']
            self.ratio.GetYaxis().SetRangeUser(yDown, yUp)
        self.ratio.GetYaxis().SetTitle("Data / Pred")
        self.ratio.GetYaxis().CenterTitle()
        self.ratio.GetYaxis().SetTitleSize(0.08)
        self.ratio.GetYaxis().SetTitleOffset(0.5)
        self.ratio.GetYaxis().SetLabelSize(0.08)
        
        # systematics
        self.ratioSyst.SetStats(0)
        self.ratioSyst.SetFillColorAlpha(ROOT.kBlack, 0.7)
        self.ratioSyst.SetFillStyle(3144)
        
    def drawLegend(self):
        self.legend.AddEntry(self.data, "Data", "lep")
        for hist in list(self.backgrounds.values())[::-1]:
            self.legend.AddEntry(hist, hist.GetName(), "f")
        self.legend.AddEntry(self.systematics, "stat+syst", "f")
        
        if self.signals:
            for hist in self.signals.values():
                self.sigLegend.AddEntry(hist, hist.GetName(), "lep")
        
    def finalize(self, textInfo=None, drawSignal=False):
        # pad up
        if self.logy: self.padUp.SetLogy()
        self.padUp.cd()
        self.data.Draw("p&hist")
        self.stack.Draw("hist&same")
        self.systematics.Draw("e2&f&same")
        self.data.Draw("p&hist&same")
        self.data.Draw("e1&same")

        if self.signals:
            for hist in self.signals.values():
                hist.Draw("hist&same")

        self.legend.Draw()
        if self.signals:
            self.sigLegend.Draw()
        
        if textInfo is None:
            self.lumi.DrawLatexNDC(0.63, 0.955, self.lumiString)
            if self.data.GetMaximum() > 10000:
                self.cms.DrawLatexNDC(0.18, 0.955, "CMS")
                self.preliminary.DrawLatexNDC(0.265, 0.955, "Preliminary")
            else:
                self.cms.DrawLatexNDC(0.115, 0.955, "CMS")
                self.preliminary.DrawLatexNDC(0.2, 0.955, "Preliminary")
        else:
            latex = TLatex()
            for text, config in textInfo.items():
                latex.SetTextSize(config[0])
                latex.SetTextFont(config[1])
                latex.DrawLatexNDC(config[2][0], config[2][1], text)
        self.padUp.RedrawAxis()
        
        # pad down
        self.padDown.cd()
        self.ratio.Draw("p&hist")
        self.ratioSyst.Draw("e2&f&same")
        self.padDown.RedrawAxis()
        
        self.cvs.cd()
        self.padUp.Draw()
        self.padDown.Draw()
        
    def draw(self):
        self.cvs.Draw()
        
    def SaveAs(self, name):
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.cvs.SaveAs(name)
