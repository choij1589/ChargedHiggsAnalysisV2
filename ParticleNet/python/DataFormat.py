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

def getMuons(evt):
    muons = []
    muons_zip = zip(evt.MuonPtColl,
                    evt.MuonEtaColl,
                    evt.MuonPhiColl,
                    evt.MuonMassColl,
                    evt.MuonChargeColl)
    for pt, eta, phi, mass, charge in muons_zip:
        thisMuon = Muon(pt, eta, phi, mass)
        thisMuon.SetCharge(charge)
        muons.append(thisMuon)
    return muons

def getElectrons(evt):
    electrons = []
    electrons_zip = zip(evt.ElectronPtColl,
                        evt.ElectronEtaColl,
                        evt.ElectronPhiColl,
                        evt.ElectronMassColl,
                        evt.ElectronChargeColl)
    for pt, eta, phi, mass, charge in electrons_zip:
        thisElectron = Electron(pt, eta, phi, mass)
        thisElectron.SetCharge(charge)
        electrons.append(thisElectron)
    return electrons

def getJets(evt):
    jets = []
    jets_zip = zip(evt.JetPtColl,
                   evt.JetEtaColl,
                   evt.JetPhiColl,
                   evt.JetMassColl,
                   evt.JetChargeColl,
                   evt.JetBtagScoreColl,
                   evt.JetIsBtaggedColl)
    for pt, eta, phi, mass, charge, btagScore, isBtagged in jets_zip:
        thisJet = Jet(pt, eta, phi, mass)
        thisJet.SetCharge(charge)
        thisJet.SetBtagScore(btagScore)
        thisJet.SetBtagging(isBtagged)
        jets.append(thisJet)

    bjets = list(filter(lambda jet: jet.IsBtagged(), jets))
    return jets, bjets
