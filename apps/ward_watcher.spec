# ward_watcher.spec — PyInstaller build spec for Ward Watcher
# Run:  pyinstaller ward_watcher.spec --clean
# Output: dist\ward_watcher\  (copy the whole folder to the target PC)

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Pull in every sub-module + data file for the three Flask-SocketIO packages
# that PyInstaller's auto-analysis tends to miss.
_sio_d, _sio_b, _sio_h   = collect_all('flask_socketio')
_sock_d, _sock_b, _sock_h = collect_all('socketio')
_eng_d, _eng_b, _eng_h    = collect_all('engineio')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=_sio_b + _sock_b + _eng_b,
    datas=(
        _sio_d + _sock_d + _eng_d
        + [
            ('templates', 'templates'),
            ('static',    'static'),
        ]
    ),
    hiddenimports=(
        _sio_h + _sock_h + _eng_h
        + [
            'engineio.async_drivers.threading',
            'pkg_resources',
        ]
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['eventlet', 'gevent'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ward_watcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ward_watcher',
)
