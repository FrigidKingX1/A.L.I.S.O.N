; A.L.I.S.O.N. - Advanced Logical Integrated Sentient Operational Network
; Phase 4: One-Click Installer & OS Integration
;
; Compiled with Inno Setup 6.2+ (ISCC.exe ALISON_Setup.iss).
; Build payload must already exist in build_dist\ (produced by the Phase 3
; build scripts: ALISON_Core.exe, ALISON_GUI.exe, ALISON_Hydrate.exe, plus
; their DLLs / data). Model weights are fetched at install time by
; ALISON_Hydrate.exe into %LOCALAPPDATA%\A.L.I.S.O.N.\models.

#define MyAppName "A.L.I.S.O.N."
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Aether Systems Architecture"
#define MyAppExeName "ALISON_GUI.exe"
#define MyAppCoreExeName "ALISON_Core.exe"
#define MyAppHydrateExeName "ALISON_Hydrate.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
OutputBaseFilename=ALISON_Setup
OutputDir=installer_output
WizardStyle=modern
WizardSmallImageFile=assets\wizard_logo.bmp
WizardImageFile=assets\wizard_logo.bmp
SetupIconFile=assets\alison_icon.ico
DisableProgramGroupPage=yes
PrivilegesRequired=admin
; Dark-mode styling for the wizard is applied via WizardStyle=modern above.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
Name: "autostart"; Description: "Start A.L.I.S.O.N. when Windows starts"; GroupDescription: "Additional icons:"

[Files]
; Entire build payload (binaries, DLLs, hydrate, appcast). The payload is
; produced by the Phase 3 build scripts into A.L.I.S.O.N\dist (..\dist from
; this installer script's location in A.L.I.S.O.N\installer).
Source: "..\dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Installer assets.
Source: "assets\alison_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\wizard_logo.bmp"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\appcast.xml"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\alison_icon.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\alison_icon.ico"

[Run]
; Phase 4: hydrate model weights during install (long, network + ~4.9 GB disk).
Filename: "{app}\{#MyAppHydrateExeName}"; Description: "Downloading A.L.I.S.O.N. model weights (this may take a while)..."; Flags: postinstall runascurrentuser; Check: ShouldRunInteractivePostInstall
; Launch the GUI once install completes.
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Registry]
; Auto-start registry key (per-user; no admin write needed).
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "ALISON"; ValueData: """{app}\{#MyAppExeName}"""; Tasks: autostart; Flags: uninsdeletevalue

[Code]
type
  TMemoryStatusEx = record
    dwLength: DWORD;
    dwMemoryLoad: DWORD;
    ullTotalPhys: Int64;
    ullAvailPhys: Int64;
    ullTotalPageFile: Int64;
    ullAvailPageFile: Int64;
    ullTotalVirtual: Int64;
    ullAvailVirtual: Int64;
    ullAvailExtendedVirtual: Int64;
  end;

function GlobalMemoryStatusEx(var lpBuffer: TMemoryStatusEx): BOOL;
  external 'GlobalMemoryStatusEx@kernel32.dll stdcall';

function GetGPUVRAMSize(): Int64;
var
  WbemServices, WbemObjectSet, WbemObject, Locator: Variant;
  i, cnt: Integer;
  v: Int64;
begin
  // Returns the largest reported dedicated VRAM across ALL video controllers.
  // Win32_VideoController.AdapterRAM is a 32-bit uint (caps at ~4 GB), so an
  // 8 GB+ card typically reports ~4 GB here -- which still clears the >= 3.5 GB
  // floor. We iterate EVERY controller and keep the max because the first
  // adapter (ItemIndex(0)) is frequently a 0-VRAM "Basic Display Adapter".
  Result := 0;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    WbemServices := Locator.ConnectServer('.', 'root\cimv2');
    WbemObjectSet := WbemServices.ExecQuery('SELECT AdapterRAM FROM Win32_VideoController');
    cnt := WbemObjectSet.Count;
    for i := 0 to cnt - 1 do
    begin
      try
        WbemObject := WbemObjectSet.ItemIndex(i);
        v := StrToInt64Def(WbemObject.AdapterRAM, 0);
        if v > Result then Result := v;
      except
        // Skip individual controllers that fail to enumerate.
      end;
    end;
  except
    Result := 0;
  end;
end;

function InitializeSetup(): Boolean;
var
  MemStatus: TMemoryStatusEx;
  TotalRAM_GB: Extended;
  VRAM_Bytes: Int64;
  VRAM_GB: Extended;
begin
  Result := True;

  // 1. Verify System Host Memory (RAM >= 16 GB)
  MemStatus.dwLength := SizeOf(MemStatus);
  if GlobalMemoryStatusEx(MemStatus) then
  begin
    TotalRAM_GB := MemStatus.ullTotalPhys / (1024 * 1024 * 1024);
    if TotalRAM_GB < 14.5 then
    begin
      if MsgBox(Format('A.L.I.S.O.N. requires at least 16 GB of System RAM.' + #13#10 +
                       'Detected RAM: %.1f GB.' + #13#10#13#10 +
                       'Do you wish to proceed anyway?', [TotalRAM_GB]),
                       mbConfirmation, MB_YESNO) = IDNO then
      begin
        Result := False;
        Exit;
      end;
    end;
  end;

  // 2. Verify GPU Video Memory (VRAM >= 8 GB)
  // Note: WMI AdapterRAM has a 4GB uint32 cap. We accept >= 3.5GB to account for this.
  VRAM_Bytes := GetGPUVRAMSize();
  if VRAM_Bytes > 0 then
  begin
    VRAM_GB := VRAM_Bytes / (1024 * 1024 * 1024);
    if VRAM_GB < 3.5 then
    begin
      if MsgBox(Format('A.L.I.S.O.N. recommends an 8 GB VRAM GPU.' + #13#10 +
                       'Detected VRAM: %.1f GB.' + #13#10#13#10 +
                       'Inference performance may be degraded. Continue installation?', [VRAM_GB]),
                       mbConfirmation, MB_YESNO) = IDNO then
      begin
        Result := False;
        Exit;
      end;
    end;
  end
  else
  begin
    MsgBox('Could not automatically read GPU VRAM (this is non-blocking).' + #13#10 +
           'A.L.I.S.O.N. will still install -- ensure your GPU has at least 8 GB VRAM for best performance.',
           mbInformation, MB_OK);
  end;
end;

function ShouldRunInteractivePostInstall(): Boolean;
begin
  // Skip the post-install model download / GUI launch when the installer runs
  // silently (automated deployments must not trigger an interactive ~4.9 GB
  // download or spawn the GUI). Interactive installs are unaffected.
  Result := not WizardSilent;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  // Placeholder for EV Code Signing of the produced ALISON_Setup.exe:
  //   SignTool sign /a /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 ALISON_Setup.exe
  // The appcast.xml channel is shipped via [Files]; host a signed installer
  // there to enable WinSparkle OTA updates.
end;
