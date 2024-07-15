from ROOT import TLorentzVector
from ROOT import TMath

# Base class for all objects
class Particle(TLorentzVector):
    def __init__(self, pt, eta, phi, mass):
        TLorentzVector.__init__(self)
        self.SetPtEtaPhiM(pt, eta, phi, mass)
        self.charge = 0
        self.btagScore = 0.
        self.isMuon = False
        self.isElectron = False
        self.isJet = False

    def Charge(self):
        return self.charge

    def BtagScore(self):
        return self.btagScore

    def MT(self, part):
        dPhi = self.DeltaPhi(part)
        return TMath.Sqrt(2*self.Pt()*part.Pt()*(1.-TMath.Cos(dPhi)))

    def IsMuon(self):
        return self.isMuon

    def IsElectron(self):
        return self.isElectron

    def IsJet(self):
        return self.isJet

# Base class for electron / muon
class Lepton(Particle):
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)

    def SetCharge(self, charge):
        self.charge = charge


class Muon(Lepton):
    def __init__(self, pt, eta, phi, mass):
        Lepton.__init__(self, pt, eta, phi, mass)
        self.isMuon = True


class Electron(Lepton):
    def __init__(self, pt, eta, phi, mass):
        Lepton.__init__(self, pt, eta, phi, mass)
        self.isElectron = True


class Jet(Particle):
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)
        self.isJet = True
        self.isBtagged = False

    def SetCharge(self, charge):
        self.charge = charge

    def SetBtagScore(self, btagScore):
        self.btagScore = btagScore

    def SetBtagging(self, isBtagged):
        self.isBtagged = isBtagged

    def IsBtagged(self):
        return self.isBtagged
