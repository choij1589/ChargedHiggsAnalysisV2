class Electron:
    def __init__(self, era, region, is_old=False):
        self.era = era
        self.region = region
        self.is_old = is_old
        self.pt = -999.
        self.scEta = -999.
        self.mvaNoIso = -999.
        self.miniRelIso = -999.
        self.sip3d = -999
        self.deltaR = -999.
        self.passMVANoIsoWP90 = False
        self.passMVANoIsoWPLoose = False
        self.nearestJetFlavour = -999
        self.genWeight = -999.
        self.mvacut = -999.
        
    def setMVACut(self): 
        if self.is_old:
            if self.era == "2016preVFP":
                if self.region == "EB1":   self.mvacut = 0.96
                elif self.region == "EB2": self.mvacut = 0.93
                else:                      self.mvacut = 0.85
            elif self.era == "2016postVFP":
                if self.region == "EB1":   self.mvacut = 0.96
                elif self.region == "EB2": self.mvacut = 0.93
                else:                      self.mvacut = 0.85
            elif self.era == "2017":
                if self.region == "EB1":   self.mvacut = 0.94
                elif self.region == "EB2": self.mvacut = 0.79
                else:                      self.mvacut = 0.5
            elif self.era == "2018":
                if self.region == "EB1":   self.mvacut = 0.94
                elif self.region == "EB2": self.mvacut = 0.79
                else:                      self.mvacut = 0.5
            else:
                raise ValueError("Invalid era")
        else:
            if self.era == "2016preVFP":
                if self.region == "EB1":   self.mvacut = 0.985
                elif self.region == "EB2": self.mvacut = 0.96
                else:                      self.mvacut = 0.85
            elif self.era == "2016postVFP":
                if self.region == "EB1":   self.mvacut = 0.985
                elif self.region == "EB2": self.mvacut = 0.96
                else:                      self.mvacut = 0.85
            elif self.era == "2017":
                if self.region == "EB1":   self.mvacut = 0.985
                elif self.region == "EB2": self.mvacut = 0.96
                else:                      self.mvacut = 0.85
            elif self.era == "2018":
                if self.region == "EB1":   self.mvacut = 0.985
                elif self.region == "EB2": self.mvacut = 0.96
                else:                      self.mvacut = 0.85
            else:
                raise ValueError("Invalid era")
    
    def setPt(self, pt):
        self.pt = pt
    
    def setPtCorr(self):
        self.ptCorr = self.pt*(1.0 + max(0., self.miniRelIso-0.1))
        
    def setScEta(self, scEta):
        self.scEta = scEta
        
    def setMVANoIso(self, mvaNoIso):
        self.mvaNoIso = mvaNoIso
    
    def setMiniRelIso(self, miniRelIso):
        self.miniRelIso = miniRelIso
    
    def setSIP3D(self, sip3d):
        self.sip3d = sip3d
        
    def setDeltaR(self, deltaR):
        self.deltaR = deltaR
        
    def setID(self, passMVANoIsoWP90, passMVANoIsoWPLoose):
        self.passMVANoIsoWP90 = passMVANoIsoWP90
        self.passMVANoIsoWPLoose = passMVANoIsoWPLoose
        
    def setNearestJetFlavour(self, nearestJetFlavour):
        self.nearestJetFlavour = nearestJetFlavour
        
    # Only required HcToWA Veto ID while skimming
    def passLooseID(self):
        if not (self.mvaNoIso > self.mvacut or self.passMVANoIsoWP90): return False
        if not self.miniRelIso < 0.4: return False
        if self.is_old:
            if not self.sip3d < 4: return False
        else:
            if not self.sip3d < 8: return False
        return True
        
    def passTightID(self):
        if not self.passMVANoIsoWP90: return False
        if not self.miniRelIso < 0.1: return False
        if not self.sip3d < 4: return False
        return True
    
    def is_valid_region(self):
        if self.region == "EB1":
            return abs(self.scEta) < 0.8
        elif self.region == "EB2":
            return abs(self.scEta) > 0.8 and abs(self.scEta) < 1.479
        elif self.region == "EE":
            return abs(self.scEta) > 1.479 and abs(self.scEta) < 2.5
        else:
            raise ValueError(f"Region {self.region} is not valid")
