using Android.App;
using Android.Content.PM;
using Avalonia;
using Avalonia.Android;

namespace LucidTensor.UI.Android;

[Activity(
    Label = "LucidTensor.UI.Android",
    Theme = "@style/MyTheme.NoActionBar",
    Icon = "@drawable/icon",
    MainLauncher = true,
    ConfigurationChanges = ConfigChanges.Orientation | ConfigChanges.ScreenSize | ConfigChanges.UiMode)]
public class MainActivity : AvaloniaMainActivity
{
}
