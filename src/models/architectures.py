import torch
import torch.nn as nn

class MicroPizzaNet(nn.Module):
    """Ultrakompaktes CNN für Pizza-Erkennung auf RP2040 mit speicherbewusster Architektur"""
    def __init__(self, num_classes=4, dropout_rate=0.2):
        super(MicroPizzaNet, self).__init__()
        
        # Verwende depthwise separable Faltungen für Speichereffizienz
        # Reduziere die Anzahl der Parameter drastisch
        
        # Erster Block: 3 -> 8 Filter
        self.block1 = nn.Sequential(
            # Standardfaltung für den ersten Layer (3 -> 8)
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 8x12x12
        )
        
        # Zweiter Block: 8 -> 16 Filter mit depthwise separable Faltung
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1) zur Kanalexpansion
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 16x6x6
        )
        
        # Feature-Extraktion abgeschlossen, jetzt kommt die Klassifikation
        
        # Global Average Pooling spart Parameter im Vergleich zu Flatten + Dense
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Ausgabe: 16x1x1
        
        # Kompakter Klassifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 16
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)  # Direkt zur Ausgabeschicht
        )
        
        # Initialisierung der Gewichte für bessere Konvergenz
        self._initialize_weights()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Verbesserte Gewichtsinitialisierung"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InvertedResidualBlock(nn.Module):
    """
    MobileNetV2-Style Inverted Residual Block mit Shortcut-Verbindung
    - Erweitert zuerst die Kanäle (Expansion)
    - Führt Depthwise Convolution durch
    - Komprimiert wieder zu Ausgabekanälen (Projection)
    - Fügt Shortcut-Verbindung hinzu, wenn Eingabe- und Ausgabedimensionen übereinstimmen
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_shortcut = stride == 1 and inp == oup
        
        # Erweiterte Kanalanzahl im Bottleneck
        hidden_dim = round(inp * expand_ratio)
        
        # Wenn Expansionsfaktor > 1, füge Expansion Layer hinzu
        layers = []
        if expand_ratio != 1:
            # 1x1 Pointwise Convolution zur Kanalexpansion
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)  # ReLU6 für Robustheit bei Quantisierung
            ])
        
        # Depthwise Convolution
        layers.extend([
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise (Projection)
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MicroPizzaNetV2(nn.Module):
    """
    Verbessertes CNN für Pizza-Erkennung mit Inverted Residual Blocks
    Optimiert für RP2040 mit effizienteren MobileNetV2-Style Blöcken
    """
    def __init__(self, num_classes=4, dropout_rate=0.2):
        super(MicroPizzaNetV2, self).__init__()
        
        # Erster Block: 3 -> 8 Filter (wie im Original)
        self.block1 = nn.Sequential(
            # Standardfaltung für den ersten Layer (3 -> 8)
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 8x12x12
        )
        
        # Zweiter Block: InvertedResidualBlock mit Expansion Factor 6 (8 -> 48 -> 16)
        self.block2 = InvertedResidualBlock(
            inp=8,             # Eingabekanäle
            oup=16,            # Ausgabekanäle
            stride=2,          # Stride (für Downsampling)
            expand_ratio=6     # Expansionsfaktor für den Bottleneck
        )
        
        # Global Average Pooling spart Parameter im Vergleich zu Flatten + Dense
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Ausgabe: 16x1x1
        
        # Kompakter Klassifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 16
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)  # Direkt zur Ausgabeschicht
        )
        
        # Initialisierung der Gewichte für bessere Konvergenz
        self._initialize_weights()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Verbesserte Gewichtsinitialisierung"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SqueezeExcitationModule(nn.Module):
    """
    Squeeze-and-Excitation Module zur Kanalgewichtung
    Implementiert 'Squeeze-and-Excitation Networks' (Hu et al., 2018)
    
    Komprimiert räumliche Dimensionen zu Kanalbeschreibungen (squeeze),
    lernt Wichtigkeit jedes Kanals (excitation) und
    gewichtet die Ausgabekanäle entsprechend.
    
    Äußerst speichereffizient für RP2040 durch geringe Parameteranzahl.
    """
    def __init__(self, channels, reduction_ratio=4):
        super(SqueezeExcitationModule, self).__init__()
        # Stelle sicher, dass reduction_ratio gültig ist
        reduced_channels = max(1, channels // reduction_ratio)
        
        # Squeeze: Global Average Pooling
        # (Geschieht in forward-Methode)
        
        # Excitation: Zwei FC-Layer mit Bottleneck
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()  # Normalisiere Gewichte auf [0,1]
        )
        
    def forward(self, x):
        # Input: [batch, channels, height, width]
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global Average Pooling -> [batch, channels, 1, 1]
        y = torch.mean(x, dim=(2, 3))  # Effiziente Alternative zu AdaptiveAvgPool2d
        
        # Excitation: FC mit Bottleneck -> [batch, channels]
        y = self.excitation(y)
        
        # Reshape für Multiplikation mit Input
        y = y.view(batch_size, channels, 1, 1)
        
        # Anwenden der Kanalgewichtung
        return x * y  # Element-weise Multiplikation


class MicroPizzaNetWithSE(nn.Module):
    """
    Erweitertes MicroPizzaNet mit Squeeze-and-Excitation Modulen zur Kanalgewichtung.
    Erhöht die Ausdruckskraft des Modells bei minimalem Parameteranstieg.
    """
    def __init__(self, num_classes=4, dropout_rate=0.2, use_se=True, se_ratio=4):
        super(MicroPizzaNetWithSE, self).__init__()
        
        self.use_se = use_se
        
        # Erster Block: 3 -> 8 Filter (wie im Original MicroPizzaNet)
        self.block1 = nn.Sequential(
            # Standardfaltung für den ersten Layer (3 -> 8)
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 8x12x12
        )
        
        # SE-Modul nach Block 1 (optional)
        if use_se:
            self.se1 = SqueezeExcitationModule(8, reduction_ratio=se_ratio)
        
        # Zweiter Block: 8 -> 16 Filter mit depthwise separable Faltung
        self.block2 = nn.Sequential(
            # Depthwise Faltung
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # Pointwise Faltung (1x1) zur Kanalexpansion
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Ausgabe: 16x6x6
        )
        
        # SE-Modul nach Block 2 (optional)
        if use_se:
            self.se2 = SqueezeExcitationModule(16, reduction_ratio=se_ratio)
        
        # Feature-Extraktion abgeschlossen, jetzt kommt die Klassifikation
        
        # Global Average Pooling spart Parameter im Vergleich zu Flatten + Dense
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Ausgabe: 16x1x1
        
        # Kompakter Klassifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 16
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)  # Direkt zur Ausgabeschicht
        )
        
        # Initialisierung der Gewichte für bessere Konvergenz
        self._initialize_weights()
    
    def forward(self, x):
        x = self.block1(x)
        
        # SE nach Block 1 (falls aktiviert)
        if self.use_se:
            x = self.se1(x)
            
        x = self.block2(x)
        
        # SE nach Block 2 (falls aktiviert)
        if self.use_se:
            x = self.se2(x)
            
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Verbesserte Gewichtsinitialisierung"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """Zählt die trainierbaren Parameter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
