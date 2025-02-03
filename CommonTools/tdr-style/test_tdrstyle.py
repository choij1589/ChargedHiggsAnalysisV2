from ROOT import TCanvas, TH1F, TF1, TLegend

class Plot(object):
    def __init__(self, name, onLeft=False):
        self.name = name
        self.canvas = TCanvas(name, name)

        self.hist = TH1F(name, "; p_{T}^{Bar} (TeV); Events / 2 TeV (10^{3})", 50, -50, 50)
        gaus1 = TF1('gaus1', 'gaus')
        gaus1.SetParameters(1, 0, 5)
        self.hist.FillRandom("gaus1", 50000)
        self.hist.Scale(0.00001)
        # self.hist.GetXaxis().SetNdivisions(5)
        # self.hist.GetXaxis().SetRangeUser(-70,70)
        self.hist.Draw()

        legend_args = (0.645, 0.79, 0.985, 0.91, '', 'NDC')
        if onLeft:
            legend_args = (0.2, 0.79, 0.540, 0.91, '', 'NDC')
        
        self.legend = TLegend(*legend_args)
        self.legend.SetFillStyle(0)
        self.legend.AddEntry(self.hist, self.hist.GetName(), "l")
        self.legend.AddEntry(self.hist, self.hist.GetName()+' b', "l")
        self.legend.Draw()

    def save(self):
        self.canvas.SaveAs('.'.join([self.name, 'png']))
        
if __name__ == '__main__':

    import tdrstyle
    from ROOT import gPad
    
    test1 = Plot('test1')
    # tdrstyle.cmsPrel(25000., 8., True)
    test1.save()
    
    test2 = Plot('test2')
    tdrstyle.cmsPrel(25000., 13., False)
    test2.save()

    test3 = Plot('test3')
    tdrstyle.cmsPrel(-1, 8., True)
    test3.save()
    
    test4 = Plot('test4', onLeft=True)
    tdrstyle.cmsPrel(-1, 8., True, onLeft=False)
    test4.save()
    
    gPad.Update()
